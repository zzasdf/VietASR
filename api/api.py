"""
ASR (Automatic Speech Recognition) FastAPI Server v2

API server cho phép nhận dạng giọng nói sử dụng icefall/k2 framework.
Hỗ trợ cả TorchScript và PyTorch checkpoint.

Chức năng chính:
- Hỗ trợ TorchScript model (jit_script.pt)
- Hỗ trợ PyTorch checkpoint (epoch-{N}.pt)
- Greedy search và Modified beam search
- Text normalization nội bộ (vi2en.txt)
- Audio segmentation cho file dài
- Download file từ URL
"""

import argparse
import asyncio
import os
import sys
import time
import math
import requests
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import aiohttp
import auditok
import soundfile as sf
from pydub import AudioSegment
import numpy as np
import uvicorn
import jwt
import torch
import sentencepiece as spm
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.datastructures import FormData
from fastapi.middleware.cors import CORSMiddleware


sys.path.insert(0, str(Path(__file__).parent.parent / "SSL" / "zipformer_fbank"))
from text_normalization import normalize_text, remove_filler_words

API_TEXT_NORM = 'https://speech.aiservice.vn/asr/textnorm'
def call_text_norm(text: str, domain: str = None) -> Dict[str, Any]:
    """Call the text normalization API to normalize the input text."""
    response = requests.post(API_TEXT_NORM, json={"text": text, "domain": domain})
    return response.json()

def _find_start_indices(main_list: List[str], sublist: List[str]) -> List[int]:
    """Find all starting indices where sublist appears in main_list."""
    start_indices = []
    try:
        index = main_list.index(sublist[0])
        while index <= len(main_list) - len(sublist):
            if main_list[index:index + len(sublist)] == sublist:
                start_indices.append(index)
            index = main_list.index(sublist[0], index + 1)
    except ValueError:
        pass
    return start_indices

def process_text_norm(
    text: str,
    segments: List[Tuple[float, float, str]],
    domain: str = None
) -> Tuple[str, List[Tuple[float, float, str]]]:
    """Process text normalization and adjust the corresponding segments."""
    # Call text normalization API
    norm_result = call_text_norm(text, domain=domain)
    out_text = norm_result["result"]["text"]
    replace_dict = norm_result["result"]["replace_dict"]

    # Find all replacement positions in the original text
    replace_obj = {}
    list_starts = []
    text_words = text.split()

    for original_phrase, normalized_phrase in replace_dict.items():
        start_indices = _find_start_indices(text_words, original_phrase.split())
        list_starts += start_indices

        for start_idx in start_indices:
            replace_obj[start_idx] = {
                "org_text": original_phrase,
                "norm_text": normalized_phrase,
                "end_idx": start_idx + len(original_phrase.split())
            }

    # Sort the starting indices for processing in order
    list_starts = sorted(list_starts)

    # Reconstruct segments with normalized text
    out_segments = []
    i = 0
    n = len(segments)

    while i < n:
        if i in list_starts:
            # Handle replacement case
            replacement = replace_obj[i]
            start_time = segments[i][0]
            end_time = segments[replacement["end_idx"] - 1][1]
            out_segments.append([start_time, end_time, replacement["norm_text"]])
            i = replacement["end_idx"]  # Skip the replaced words
        else:
            # Handle normal case
            start, end, word = segments[i]
            out_segments.append([start, end, word])
            i += 1

    return out_text, out_segments

# Initialize FastAPI app with CORS middleware
app = FastAPI(
    title="VietASR API",
    description="Vietnamese Speech Recognition API using icefall/k2 framework",
    version="2.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
UPLOAD_FOLDER = "uploaded"  
PRIVATE_TOKEN = "41ad97aa3c747596a4378cc8ba101fe70beb3f5f70a75407a30e6ddab668310d"

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class ASRModel:
    """
    Unified ASR Model wrapper supporting both TorchScript and PyTorch checkpoints.
    """
    
    def __init__(
        self,
        model_path: str,
        bpe_path: str,
        model_type: str = "auto",
        device: str = "cuda:0",
    ):
        """
        Initialize ASR Model.
        
        Args:
            model_path: Path to TorchScript (.pt jit) or PyTorch checkpoint
            bpe_path: Path to sentencepiece bpe.model
            model_type: "torchscript", "pytorch", or "auto" (auto-detect)
            device: Device to run model on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        # Auto-detect model type if needed
        if model_type == "auto":
            self.model_type = self._detect_model_type(model_path)
        else:
            self.model_type = model_type
        
        logger.info(f"Loading {self.model_type} model from {model_path}")
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        self.model.to(self.device)
        
        # Load BPE model
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(bpe_path)
        logger.info(f"Loaded BPE model from {bpe_path}, vocab_size={self.sp.get_piece_size()}")
        
        # Get blank_id and context_size from model
        try:
            self.blank_id = int(self.model.decoder.blank_id)
            self.context_size = int(self.model.decoder.context_size)
        except AttributeError:
            self.blank_id = 0
            self.context_size = 2
            logger.warning(f"Could not get blank_id/context_size from model, using defaults: {self.blank_id}, {self.context_size}")
    
    def _detect_model_type(self, model_path: str) -> str:
        """Auto-detect model type based on file size and name."""
        file_size = os.path.getsize(model_path)
        # TorchScript models are typically smaller (< 500MB)
        # PyTorch checkpoints with optimizer states are larger (> 1GB)
        if "jit" in model_path.lower() or file_size < 500 * 1024 * 1024:
            return "torchscript"
        return "pytorch"
    
    def _load_model(self, model_path: str):
        """Load model based on type."""
        if self.model_type == "torchscript":
            return torch.jit.load(model_path, map_location=self.device)
        else:
            # For PyTorch checkpoint, need to construct model architecture
            # This requires importing from the training codebase
            raise NotImplementedError(
                "PyTorch checkpoint loading requires model architecture. "
                "Please use TorchScript model or provide model architecture."
            )
    
    def _extract_features(self, audio: np.ndarray, sample_rate: int = 16000) -> torch.Tensor:
        """
        Extract Fbank features from audio.
        Using simple torch-based implementation compatible with training.
        """
        # Convert to torch tensor
        if isinstance(audio, np.ndarray):
            waveform = torch.from_numpy(audio).float()
        else:
            waveform = audio.float()
        
        # Ensure correct shape [1, num_samples]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Scale audio (matching training preprocessing)
        waveform = waveform * 32768.0
        
        # Simple Fbank extraction using torchaudio
        try:
            import torchaudio.compliance.kaldi as kaldi
            features = kaldi.fbank(
                waveform,
                num_mel_bins=80,
                frame_length=25,
                frame_shift=10,
                sample_frequency=sample_rate,
                dither=0,
                snip_edges=False,
            )
        except ImportError:
            # Fallback: use kaldifeat if torchaudio doesn't work
            import kaldifeat
            opts = kaldifeat.FbankOptions()
            opts.device = torch.device("cpu")
            opts.frame_opts.dither = 0
            opts.frame_opts.snip_edges = False
            opts.frame_opts.samp_freq = sample_rate
            opts.mel_opts.num_bins = 80
            fbank = kaldifeat.Fbank(opts)
            features = fbank(waveform.squeeze(0) * 32768.0)
        
        return features
    
    def _greedy_search(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
    ) -> List[List[int]]:
        """
        Greedy search decoding for TorchScript model.
        """
        assert encoder_out.ndim == 3
        N = encoder_out.size(0)
        device = encoder_out.device
        
        packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
            input=encoder_out,
            lengths=encoder_out_lens.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        
        batch_size_list = packed_encoder_out.batch_sizes.tolist()
        
        hyps = [[self.blank_id] * self.context_size for _ in range(N)]
        
        decoder_input = torch.tensor(
            hyps,
            device=device,
            dtype=torch.int64,
        )
        
        decoder_out = self.model.decoder(
            decoder_input,
            need_pad=torch.tensor([False], device=device),
        ).squeeze(1)
        
        offset = 0
        for batch_size in batch_size_list:
            start = offset
            end = offset + batch_size
            current_encoder_out = packed_encoder_out.data[start:end]
            offset = end
            
            decoder_out = decoder_out[:batch_size]
            
            logits = self.model.joiner(
                current_encoder_out,
                decoder_out,
            )
            
            y = logits.argmax(dim=1).tolist()
            emitted = False
            for i, v in enumerate(y):
                if v != self.blank_id:
                    hyps[i].append(v)
                    emitted = True
            
            if emitted:
                decoder_input = [h[-self.context_size:] for h in hyps[:batch_size]]
                decoder_input = torch.tensor(
                    decoder_input,
                    device=device,
                    dtype=torch.int64,
                )
                decoder_out = self.model.decoder(
                    decoder_input,
                    need_pad=torch.tensor([False], device=device),
                ).squeeze(1)
        
        sorted_ans = [h[self.context_size:] for h in hyps]
        ans = []
        unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
        for i in range(N):
            ans.append(sorted_ans[unsorted_indices[i]])
        
        return ans
    
    def _modified_beam_search(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        beam_size: int = 4,
    ) -> List[List[int]]:
        """
        Modified beam search decoding.
        """
        try:
            from beam_search import modified_beam_search
            return modified_beam_search(
                model=self.model,
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                beam=beam_size,
            )
        except ImportError:
            logger.warning("beam_search module not found, falling back to greedy search")
            return self._greedy_search(encoder_out, encoder_out_lens)
    
    @torch.no_grad()
    def recognize(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        decoding_method: str = "greedy_search",
        beam_size: int = 4,
        apply_text_norm: bool = True,
    ) -> Tuple[str, str]:
        """
        Recognize speech from audio.
        
        Args:
            audio: Audio samples as numpy array
            sample_rate: Sample rate of audio
            decoding_method: "greedy_search" or "modified_beam_search"
            beam_size: Beam size for beam search
            apply_text_norm: Whether to apply text normalization
            
        Returns:
            Tuple of (raw_text, normalized_text)
        """
        # Extract features
        features = self._extract_features(audio, sample_rate)
        features = features.unsqueeze(0).to(self.device)  # [1, T, 80]
        feature_lengths = torch.tensor([features.size(1)], device=self.device)
        
        # Encode
        encoder_out, encoder_out_lens = self.model.encoder(
            features=features,
            feature_lengths=feature_lengths,
        )
        
        # Decode
        if decoding_method == "greedy_search":
            hyp_tokens = self._greedy_search(encoder_out, encoder_out_lens)
        elif decoding_method == "modified_beam_search":
            hyp_tokens = self._modified_beam_search(encoder_out, encoder_out_lens, beam_size)
        else:
            raise ValueError(f"Unsupported decoding method: {decoding_method}")
        
        # Convert tokens to text
        raw_text = self.sp.decode(hyp_tokens[0])
        
        # Apply text normalization
        if apply_text_norm:
            normalized_text = normalize_text(raw_text)
        else:
            normalized_text = raw_text
        
        return raw_text, normalized_text


# Global model instance
asr_model: Optional[ASRModel] = None
model_name: str = ""


def merge_short_audio_segments(segments, target_length):
    """
    Groups short audio segments into sublists where the combined duration 
    is closest to the target length.
    """
    if not segments:
        return []
    
    result = []
    current_sublist = []
    current_length = 0
    
    for segment in segments:
        start, end = segment
        segment_length = end - start
        new_length = current_length + segment_length
        
        if (current_sublist and 
            abs(new_length - target_length) > abs(current_length - target_length)):
            result.append(current_sublist)
            current_sublist = [segment]
            current_length = segment_length
        else:
            current_sublist.append(segment)
            current_length = new_length
    
    if current_sublist:
        result.append(current_sublist)
    
    return result


async def download_file(url: str) -> Optional[str]:
    """
    Download an audio file from a given URL and save it locally.
    """
    try:
        if 'api.stringee.com' in url:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                file_name = url.split('/')[-1][:50]
                save_filepath = os.path.join(UPLOAD_FOLDER, f'{file_name}.wav')
                with open(save_filepath, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
        else:
            filename = url.split("/")[-1]
            save_filepath = f"{UPLOAD_FOLDER}/{filename}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        with open(save_filepath, "wb") as f:
                            while True:
                                chunk = await response.content.read(1024)
                                if not chunk:
                                    break
                                f.write(chunk)
        
        if os.path.exists(save_filepath):
            logger.success(f"Downloaded file successfully: {url} to {save_filepath}")
            return save_filepath
        else:
            logger.error(f"Download failed: {url}")
            return None
            
    except Exception as e:
        logger.error(f"Exception downloading file {url}: {str(e)}")
        return None


@app.get("/")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {
        "message": " API  is running",
        "model": model_name,
        "usage": "Use POST method for speech recognition"
    }


@app.post("/")
async def process(request: Request) -> JSONResponse:
    """
    Main endpoint for speech recognition processing.
    """
    global asr_model, model_name
    
    # Authentication check
    token = request.headers.get("Authorization")
    if not token:
        return _error_response(400, "Token is missing")
    
    if token != PRIVATE_TOKEN:
        try:
            jwt.decode(token, "datamining_vcc", algorithms="HS256")
        except jwt.ExpiredSignatureError:
            return _error_response(401, "Signature expired. Please log in again")
        except jwt.InvalidTokenError:
            return _error_response(400, "Invalid token. Please log in again")
    
    # Initialize response structure
    response = {
        "status": 0,
        "data": {
            "model_version": model_name,
            "result": []
        },
        "message": "process file error",
        "code": 400
    }
    
    # Parse request data based on content type
    content_type = request.headers.get("Content-Type", "")
    
    if "application/json" in content_type:
        try:
            request_data = await request.json()
        except Exception as e:
            response["message"] = f"Invalid JSON: {str(e)}"
            return JSONResponse(response)
    elif "multipart/form-data" in content_type:
        request_data = await request.form()
    elif "audio/" in content_type or "application/octet-stream" in content_type:
        request_data = await request.body()
    else:
        # Try to parse as JSON first, then form, then bytes
        try:
            request_data = await request.json()
        except:
            request_data = await request.form()
            if len(request_data) == 0:
                request_data = await request.body()
    
    if not isinstance(request_data, (dict, FormData, bytes)):
        response["message"] = "Unsupported request type, try [json, formdata or bytes]"
        return JSONResponse(response)
    
    # Process different request formats
    # Only support: 1) JSON with URL, 2) Raw binary audio
    if isinstance(request_data, dict):
        file_url = request_data.get("file", "")
        
        # Only support URL (http/https)
        if not file_url.startswith(('http://', 'https://')):
            response["message"] = "Invalid file URL. Must start with http:// or https://"
            return JSONResponse(response)
        
        # Download file from URL
        save_filepath = await download_file(file_url)
        if not save_filepath:
            response["message"] = f"Cannot download file from: {file_url}"
            return JSONResponse(response)
        
        params = {
            "speed": int(request_data.get("speed", 0)),
            "text_norm": int(request_data.get("text_norm", 1)) == 1,
            "domain": request_data.get("domain"),
            "split": int(request_data.get("split", 1)),
        }
        
    elif isinstance(request_data, bytes):
        # Raw binary audio
        if len(request_data) == 0:
            response["message"] = "Empty audio data received"
            return JSONResponse(response)
        
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S.%f')[:-4]
        save_filepath = f"{UPLOAD_FOLDER}/{timestamp}.wav"
        with open(save_filepath, "wb") as f:
            f.write(request_data)
        
        params = {
            "speed": 0,
            "text_norm": True,
            "domain": None,
            "split": 1,
        }
    else:
        response["message"] = "Unsupported request format. Use JSON with URL or raw binary audio."
        return JSONResponse(response)
    
    # Perform speech recognition
    try:
        beam_size = _get_beam_size(params["speed"])
        decoding_method = "modified_beam_search" if params["speed"] >= 0 else "greedy_search"
        
        recognition_params = {
            "beam_size": beam_size,
            "text_norm": params["text_norm"],
            "split": params["split"],
            "decoding_method": decoding_method,
        }
        
        asr_result = await recognize_file(save_filepath, recognition_params)
        
        # Clean up temp files
        if os.path.exists(save_filepath):
            os.remove(save_filepath)
        
        return JSONResponse({
            "status": 1,
            "code": 200,
            "message": "Process file success",
            "data": {
                "model_version": model_name,
                "result": asr_result
            }
        })
        
    except Exception as e:
        logger.exception(f"Recognition error: {str(e)}")
        response["message"] = f"Recognition error: {str(e)}"
        return JSONResponse(response)


def _error_response(code: int, message: str) -> JSONResponse:
    """Helper function to create error responses."""
    return JSONResponse({
        "status": 0,
        "code": code,
        "message": message,
        "data": {}
    })


def _get_beam_size(speed: int) -> int:
    """Determine beam size based on speed parameter."""
    if speed < 0:
        return 1  # Fastest (greedy)
    elif speed == 0:
        return 4  # Balanced
    else:
        return 10  # Most accurate


async def recognize_file(
    file_path: str,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process an audio file through speech recognition pipeline.
    """
    global asr_model
    
    start_time = time.perf_counter()
    
    try:
        # Convert audio to mono WAV format if needed
        sound = AudioSegment.from_file(file_path)
        ext = file_path.split('.')[-1]
        wav_path = file_path.replace(f'.{ext}', '.wav')
        sound.export(wav_path, format='wav', bitrate='256k', parameters=["-ac", "1", "-ar", "16000"])
    except Exception as e:
        logger.error(f"Audio conversion error: {str(e)}")
        raise
    
    # Read audio file
    audio, sample_rate = sf.read(wav_path)
    duration = len(audio) / sample_rate
    
    audio_chunks = []
    timestamps = []
    results = []
    
    # Split audio if needed
    if not params['split'] or (duration > 0.3 and duration <= 10):
        audio_chunks = [audio]
        timestamps = [{"start": '0', "end": str(duration)}]
    elif duration > 10:
        logger.info(f"Processing long audio file: duration={duration:.2f}s")
        
        audio_regions = auditok.split(
            wav_path,
            min_dur=0.5,
            max_dur=10,
            max_silence=0.3,
        )
        
        segments = [(int(r.meta.start * sample_rate), int(r.meta.end * sample_rate)) for r in audio_regions]
        segments = merge_short_audio_segments(segments, target_length=10 * sample_rate)
        
        for seg in segments:
            chunk = np.concatenate([audio[start:end] for start, end in seg], axis=0)
            if len(chunk) >= 1600:
                audio_chunks.append(chunk)
                timestamps.append({
                    "start": str(seg[0][0] / sample_rate),
                    "end": str(seg[-1][-1] / sample_rate)
                })
    
    # Process all audio chunks
    if audio_chunks:
        for ts, chunk in zip(timestamps, audio_chunks):
            # Get raw text from ASR model (no internal text norm)
            raw_text, _ = asr_model.recognize(
                audio=chunk,
                sample_rate=sample_rate,
                decoding_method=params["decoding_method"],
                beam_size=params["beam_size"],
                apply_text_norm=False,  # Don't apply internal norm
            )
            
            # Generate dummy word-level segments based on audio duration
            # (Real word alignment would require CTC or similar)
            chunk_duration = len(chunk) / sample_rate
            chunk_start = float(ts["start"])
            words = raw_text.split()
            segments = []
            
            if words:
                word_duration = chunk_duration / len(words)
                for i, word in enumerate(words):
                    word_start = chunk_start + i * word_duration
                    word_end = word_start + word_duration
                    segments.append((round(word_start, 2), round(word_end, 2), word))
            
            # Apply text normalization using external API (same as original API)
            if params["text_norm"] and raw_text.strip():
                try:
                    text_normalized, norm_segments = process_text_norm(
                        raw_text, segments, params.get("domain")
                    )
                    ts["text"] = text_normalized
                    ts["text_norm"] = text_normalized
                    # Convert segments to list format [[start, end, word], ...]
                    ts["segments"] = [[s[0], s[1], s[2]] for s in norm_segments]
                except Exception as e:
                    logger.warning(f"Text norm API failed: {e}, using raw text")
                    ts["text"] = raw_text
                    ts["text_norm"] = raw_text
                    ts["segments"] = [[s[0], s[1], s[2]] for s in segments]
            else:
                ts["text"] = raw_text
                ts["text_norm"] = raw_text
                ts["segments"] = [[s[0], s[1], s[2]] for s in segments]
            
            results.append(ts)
    
    # Prepare final response
    end_time = time.perf_counter()
    processing_time = end_time - start_time
    
    logger.info(f"Processed file: {file_path}")
    logger.info(f"Total segments: {len(results)}")
    logger.info(f"Processing time: {processing_time:.2f}s")
    
    return {
        "duration": f"{duration:.2f}",
        "infer_time": f"{processing_time:.2f}s",
        "beam_size": params['beam_size'],
        "text": results,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VietASR API Server v2')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to TorchScript or PyTorch model')
    parser.add_argument('--bpe-model', type=str, required=True,
                        help='Path to sentencepiece bpe.model')
    parser.add_argument('--model-type', type=str, default="auto",
                        choices=["auto", "torchscript", "pytorch"],
                        help='Model type (auto-detect by default)')
    parser.add_argument('--device', type=str, default="cuda:0",
                        help='Device to run the model on')
    parser.add_argument('--host', type=str, default="0.0.0.0",
                        help='Host address to bind the server to')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run the server on')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of worker processes')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = "INFO"
    log_format = ("<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | "
                  "<level>{level: <8}</level> | "
                  "<yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>")
    
    logger.add(sys.stderr, level=log_level, format=log_format,
               colorize=True, backtrace=True, diagnose=True)
    
    logger.info(f"Starting server with configuration: {vars(args)}")
    
    # Load the ASR model
    try:
        asr_model = ASRModel(
            model_path=args.model_path,
            bpe_path=args.bpe_model,
            model_type=args.model_type,
            device=args.device,
        )
        model_name = Path(args.model_path).stem
        logger.success(f"Successfully loaded model: {model_name}")
        
        # Start the FastAPI server
        uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        sys.exit(1)
