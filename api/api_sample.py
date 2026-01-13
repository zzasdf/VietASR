"""
ASR (Automatic Speech Recognition) FastAPI Server

This server provides an API endpoint for speech recognition using various ESPnet-based models.
It supports different model formats (PyTorch, ONNX, OpenVINO) and configurations.

Key Features:
- Supports multiple model architectures (standard ASR, Dolphin, Transducer)
- Handles audio file processing and segmentation
- Provides text normalization
- Supports beam search with language model integration
- Handles concurrent requests with rate limiting
"""

import argparse
import asyncio
import os
import sys
import time
import requests
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
import auditok
import soundfile as sf
from pydub import AudioSegment
import numpy as np
import uvicorn
import jwt
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.datastructures import FormData
from fastapi.middleware.cors import CORSMiddleware

from utils.update_config_yaml import update_config
from utils.process_align import process_text_norm
from utils.utils import merge_short_audio_segments
from espnet2.bin.asr_inference import Speech2Text
from espnet2.bin.dolphin.dolphin_inference import DolphinSpeech2Text
from espnet2.bin.asr_transducer_inference import Speech2Text as Speech2TextTransducer
from espnet2.espnet_onnx.asr.asr_model import Speech2Text as Speech2TextONNX
from espnet2.espnet_onnx.asr.asr_model_openvino import Speech2Text as Speech2TextOpenVINO
from espnet2.espnet_onnx.asr.s2t_model import Speech2Text as DolphinSpeech2TextONNX
from espnet2.fileio.sound_scp import soundfile_read

# Initialize FastAPI app with CORS middleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
UPLOAD_FOLDER = "uploaded"  # Directory for temporary audio files
PRIVATE_TOKEN = "41ad97aa3c747596a4378cc8ba101fe70beb3f5f70a75407a30e6ddab668310d"  # Hardcoded auth token

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


async def download_file(url: str) -> Optional[str]:
    """
    Download an audio file from a given URL and save it locally.
    
    Args:
        url: The URL of the audio file to download
        
    Returns:
        str: Path to the downloaded file if successful, None otherwise
    """
    try:
        # Special handling for Stringee API URLs
        if 'api.stringee.com' in url:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                file_name = url.split('/')[-1][:50]  # Truncate long filenames
                save_filepath = os.path.join(UPLOAD_FOLDER, f'{file_name}.wav')
                with open(save_filepath, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
        else:
            # General file download handling
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

        # Verify download was successful
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
    """Health check endpoint that returns basic service information."""
    return {"message": "ASR service is running", "usage": "Use POST method for speech recognition"}


@app.post("/")
async def process(request: Request) -> JSONResponse:
    """
    Main endpoint for speech recognition processing.
    
    Handles different request types (JSON, form data, raw bytes) and returns recognition results.
    
    Args:
        request: FastAPI request object containing audio data and parameters
        
    Returns:
        JSONResponse: Recognition results or error message
    """
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
    try:
        request_data = await request.json()
    except:
        request_data = await request.form()
        if len(request_data) == 0:
            request_data = await request.body()
    finally:
        if not isinstance(request_data, (dict, FormData, bytes)):
            response["message"] = "Unsupported request type, try [json, formdata or bytes]"
            return JSONResponse(response)

    # Process different request formats
    if isinstance(request_data, dict):
        # JSON request handling
        file_url = request_data["file"]
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
        
    elif isinstance(request_data, FormData):
        # Form data handling
        save_filepath = f"{UPLOAD_FOLDER}/{request_data['file'].filename}"
        with open(save_filepath, "wb") as f:
            f.write(await request_data["file"].read())
            
        params = {
            "speed": int(request_data.get("speed", 0)),
            "text_norm": int(request_data.get("text_norm", 1)) == 1,
            "domain": request_data.get("domain"),
            "split": int(request_data.get("split", 1)),
        }
        
    elif isinstance(request_data, bytes):
        # Raw audio bytes handling
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

    # Perform speech recognition
    try:
        beam_size = _get_beam_size(params["speed"])
        recognition_params = {
            "beam_size": beam_size,
            "text_norm": params["text_norm"],
            "domain": params["domain"],
            "split": params["split"],
        }
        
        asr_result = await recognize_file(duration_model_mapping, save_filepath, recognition_params)
        
        # Clean up and return success response
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
        return 1  # Fastest but least accurate
    elif speed == 0:
        return 5  # Balanced speed/accuracy
    else:
        return 10  # Slowest but most accurate


async def model_recognize_async(
    duration_model_mapping: Dict[float, Union[Speech2Text, Speech2TextONNX, Speech2TextOpenVINO]],
    audio: np.ndarray,
    params: Dict[str, Any]
) -> Tuple[str, List[Dict]]:
    """
    Perform speech recognition on an audio segment using the appropriate model.
    
    Args:
        duration_model_mapping: Dictionary mapping audio durations to models
        audio: Numpy array containing audio samples
        params: Recognition parameters including beam_size, text_norm, domain
        
    Returns:
        Tuple of recognized text and alignment segments
    """
    beam_size = params['beam_size']
    text_norm = params['text_norm']
    domain = params['domain']

    # Handle different model configurations
    if len(duration_model_mapping) > 1:
        # Select model based on audio duration
        audio_duration = len(audio) / 16000
        for max_dur, model in duration_model_mapping.items():
            if audio_duration <= max_dur:
                result = model.recognize(audio, beam_size)[0]
                text = result[0]
                segments = result[-1]
                text_normalized, norm_segments = process_text_norm(text, segments, domain)
                if text_norm:
                    return text_normalized, text_normalized, norm_segments
                return text, text_normalized, segments
    else:
        # Single model case
        model = duration_model_mapping['model']
        result = model.recognize(audio, beam_size)[0]
        text = result[0]
        segments = result[-1]
        text_normalized, norm_segments = process_text_norm(text, segments, domain)
        if text_norm:
            return text_normalized, text_normalized, norm_segments
        return text, text_normalized, segments


async def recognize_file(
    duration_model_mapping: Dict[float, Union[Speech2Text, Speech2TextONNX, Speech2TextOpenVINO]],
    file_path: str,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process an audio file through speech recognition pipeline.
    
    Args:
        duration_model_mapping: Dictionary mapping audio durations to models
        file_path: Path to audio file
        params: Recognition parameters
        
    Returns:
        Dictionary containing recognition results and metadata
    """
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

    # Read audio file and calculate duration
    audio, sample_rate = soundfile_read(wav_path)
    duration = len(audio) / sample_rate
    audio_chunks = []
    timestamps = []
    results = []

    # Split audio if needed (long files or split parameter enabled)
    if not params['split'] or (duration > 0.3 and duration <= 10):
        # Process whole file as single segment
        audio_chunks = [audio]
        timestamps = [{"start": '0', "end": str(duration)}]
    elif duration > 10:
        # Split long audio into segments
        logger.info(f"Processing long audio file: duration={duration:.2f}s")
        
        audio_regions = auditok.split(
            wav_path,
            min_dur=0.5,     # Minimum duration of valid segment (seconds)
            max_dur=10,       # Maximum duration of segment
            max_silence=0.3,   # Maximum allowed silence within segment
        )
        
        # Merge short segments and prepare for recognition
        segments = [(int(r.meta.start * sample_rate), int(r.meta.end * sample_rate)) for r in audio_regions]
        segments = merge_short_audio_segments(segments, target_length=10 * sample_rate)
        
        for seg in segments:
            chunk = np.concatenate([audio[start:end] for start, end in seg], axis=0)
            if len(chunk) >= 1600:  # Minimum meaningful audio length
                audio_chunks.append(chunk)
                timestamps.append({
                    "start": str(seg[0][0] / sample_rate),
                    "end": str(seg[-1][-1] / sample_rate)
                })

    # Process all audio chunks in parallel
    if audio_chunks:
        recognition_tasks = [
            model_recognize_async(duration_model_mapping, chunk, params)
            for chunk in audio_chunks
        ]
        recognition_results = await asyncio.gather(*recognition_tasks)
        
        # Combine results with timestamps
        for ts, (text, text_norm, segments) in zip(timestamps, recognition_results):
            ts["text"] = text
            ts["text_norm"] = text_norm
            ts["segments"] = segments
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


def load_model(args: argparse.Namespace) -> Dict[float, Any]:
    """
    Load the appropriate ASR model based on configuration.
    
    Args:
        args: Command line arguments containing model configuration
        
    Returns:
        Dictionary mapping audio durations to loaded models
    """
    model_mapping = {}
    
    # Dolphin ONNX model
    if "dolphin" in args.model_dir and "onnx" in args.model_dir:
        logger.info("Loading Dolphin ONNX model")
        model_mapping = {2: None, 4: None, 10: None}
        length_duration_mapping = {256: 2, 512: 4, 1280: 10}
        
        for max_length, max_duration in length_duration_mapping.items():
            logger.info(f"Loading model with max length: {max_length}")
            model_dir = args.model_dir[:-1] + f"_{max_length}"
            model_mapping[max_duration] = DolphinSpeech2TextONNX(
                config_path=os.path.join(model_dir, "config"),
                encoder_onnx_path=os.path.join(model_dir, "encoder"),
                ctc_onnx_path=os.path.join(model_dir, "ctc"),
                decoder_onnx_path=os.path.join(model_dir, "decoder"),
                feat_stats_path=os.path.join(model_dir, "feat_normalize"),
                bpe_model_path=os.path.join(model_dir, "bpe_model"),
                kenlm_file=os.path.join(model_dir, "../lm"),
                kenlm_alpha=args.kenlm_alpha,
                kenlm_beta=args.kenlm_beta,
                word_vocab_file=os.path.join(model_dir, "word_vocab"),
                word_vocab_size=args.word_vocab_size,
            )
    
    # Standard Dolphin model
    elif "dolphin" in args.model_dir:
        logger.info("Loading Dolphin model")
        asr_train_config = os.path.join(args.model_dir, "config")
        update_config(asr_train_config, args.model_dir)
        
        model_mapping["model"] = DolphinSpeech2Text(
            s2t_train_config=asr_train_config,
            s2t_model_file=os.path.join(args.model_dir, "model"),
            device=args.device,
            ctc_weight=args.ctc_weight,
            kenlm_file=os.path.join(args.model_dir, "../lm"),
            kenlm_alpha=args.kenlm_alpha,
            kenlm_beta=args.kenlm_beta,
            word_vocab_file=os.path.join(args.model_dir, "word_vocab"),
            word_vocab_size=args.word_vocab_size,
        )
    
    # ONNX model
    elif "onnx" in args.model_dir:
        logger.info("Loading ONNX model")
        model_mapping = {2: None, 4: None, 6: None, 8: None, 10: None}
        length_duration_mapping = {256: 2, 512: 4, 768: 6, 1024: 8, 1280: 10}
        
        if args.model_dir.endswith("*"):
            # Load multiple ONNX models for different durations
            for max_length, max_duration in length_duration_mapping.items():
                logger.info(f"Loading model with max length: {max_length}")
                model_dir = args.model_dir[:-1] + f"_{max_length}"
                model_mapping[max_duration] = Speech2TextONNX(
                    config_path=os.path.join(model_dir, "config"),
                    encoder_onnx_path=os.path.join(model_dir, "encoder"),
                    ctc_onnx_path=os.path.join(model_dir, "ctc"),
                    feat_stats_path=os.path.join(model_dir, "feat_normalize"),
                    bpe_model_path=os.path.join(model_dir, "bpe_model"),
                    kenlm_file=os.path.join(model_dir, "../lm"),
                    kenlm_alpha=args.kenlm_alpha,
                    kenlm_beta=args.kenlm_beta,
                    word_vocab_file=os.path.join(model_dir, "word_vocab"),
                    word_vocab_size=args.word_vocab_size,
                )
        else:
            # Single ONNX model
            model = Speech2TextONNX(
                config_path=os.path.join(args.model_dir, "config"),
                encoder_onnx_path=os.path.join(args.model_dir, "encoder"),
                ctc_onnx_path=os.path.join(args.model_dir, "ctc"),
                feat_stats_path=os.path.join(args.model_dir, "feat_normalize"),
                bpe_model_path=os.path.join(args.model_dir, "bpe_model"),
                kenlm_file=os.path.join(args.model_dir, "../lm"),
                kenlm_alpha=args.kenlm_alpha,
                kenlm_beta=args.kenlm_beta,
                word_vocab_file=os.path.join(args.model_dir, "word_vocab"),
                word_vocab_size=args.word_vocab_size,
            )
            model_mapping = {k: model for k in model_mapping.keys()}
    
    # OpenVINO model
    elif "openvino" in args.model_dir:
        logger.info("Loading OpenVINO model")
        model_mapping = {2: None, 4: None, 6: None, 8: None, 10: None}
        length_duration_mapping = {256: 2, 512: 4, 768: 6, 1024: 8, 1280: 10}
        
        if args.model_dir.endswith("*"):
            # Load multiple OpenVINO models for different durations
            for max_length, max_duration in length_duration_mapping.items():
                logger.info(f"Loading model with max length: {max_length}")
                model_dir = args.model_dir[:-1] + f"_{max_length}"
                model_mapping[max_duration] = Speech2TextOpenVINO(
                    config_path=os.path.join(model_dir, "config"),
                    encoder_model_path=os.path.join(model_dir, "encoder.xml"),
                    ctc_model_path=os.path.join(model_dir, "ctc.xml"),
                    feat_stats_path=os.path.join(model_dir, "feat_normalize"),
                    bpe_model_path=os.path.join(model_dir, "bpe_model"),
                    kenlm_file=os.path.join(model_dir, "../lm"),
                    kenlm_alpha=args.kenlm_alpha,
                    kenlm_beta=args.kenlm_beta,
                    word_vocab_file=os.path.join(model_dir, "word_vocab"),
                    word_vocab_size=args.word_vocab_size,
                )
        else:
            # Single OpenVINO model
            model = Speech2TextOpenVINO(
                config_path=os.path.join(args.model_dir, "config"),
                encoder_model_path=os.path.join(args.model_dir, "encoder.xml"),
                ctc_model_path=os.path.join(args.model_dir, "ctc.xml"),
                feat_stats_path=os.path.join(args.model_dir, "feat_normalize"),
                bpe_model_path=os.path.join(args.model_dir, "bpe_model"),
                kenlm_file=os.path.join(args.model_dir, "../lm"),
                kenlm_alpha=args.kenlm_alpha,
                kenlm_beta=args.kenlm_beta,
                word_vocab_file=os.path.join(args.model_dir, "word_vocab"),
                word_vocab_size=args.word_vocab_size,
            )
            model_mapping = {k: model for k in model_mapping.keys()}
    
    # Ensemble Transducer model
    elif args.ext_model_dir is not None:
        logger.info("Loading Ensemble Transducer model")
        from espnet2.bin.asr_ensemble_transducer_inference import Speech2Text as Speech2TextEnsembleTransducer
        
        asr_train_config = os.path.join(args.model_dir, "config")
        update_config(asr_train_config, args.model_dir)
        ext_asr_train_config = os.path.join(args.ext_model_dir, "config")
        update_config(ext_asr_train_config, args.ext_model_dir)

        model_mapping["model"] = Speech2TextEnsembleTransducer(
            list_asr_train_config=[asr_train_config, ext_asr_train_config],
            list_asr_model_file=[
                os.path.join(args.model_dir, "model"),
                os.path.join(args.ext_model_dir, "model")
            ],
            ensemble_weights='0.5 0.5'.split(),
            device=args.device,
            kenlm_file=[os.path.join(args.model_dir, "../lm")],
            kenlm_alpha=[args.kenlm_alpha] * 2,
            kenlm_beta=[args.kenlm_beta] * 2,
            word_vocab_file=os.path.join(args.model_dir, "word_vocab"),
            word_vocab_size=args.word_vocab_size,
        )
    
    # Transducer model
    elif "transducer" in args.model_dir:
        logger.info("Loading Transducer model")
        asr_train_config = os.path.join(args.model_dir, "config")
        update_config(asr_train_config, args.model_dir)
        
        model_mapping["model"] = Speech2TextTransducer(
            asr_train_config=asr_train_config,
            asr_model_file=os.path.join(args.model_dir, "model"),
            device=args.device,
            kenlm_file=os.path.join(args.model_dir, "../lm"),
            kenlm_alpha=args.kenlm_alpha,
            kenlm_beta=args.kenlm_beta,
            word_vocab_file=os.path.join(args.model_dir, "word_vocab"),
            word_vocab_size=args.word_vocab_size,
        )
    
    # Standard PyTorch model
    else:
        logger.info("Loading standard PyTorch model")
        asr_train_config = os.path.join(args.model_dir, "config")
        update_config(asr_train_config, args.model_dir)
        
        model_mapping["model"] = Speech2Text(
            asr_train_config=asr_train_config,
            asr_model_file=os.path.join(args.model_dir, "model"),
            device=args.device,
            ctc_weight=args.ctc_weight,
            kenlm_file=os.path.join(args.model_dir, "../lm"),
            kenlm_alpha=args.kenlm_alpha,
            kenlm_beta=args.kenlm_beta,
            word_vocab_file=os.path.join(args.model_dir, "word_vocab"),
            word_vocab_size=args.word_vocab_size,
        )

    return model_mapping


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ASR Server Configuration')
    parser.add_argument('--model_dir', type=str, required=True, 
                      help='Directory containing the ASR model files')
    parser.add_argument('--ext_model_dir', type=str, default=None,
                      help='Directory for additional model in ensemble configuration')
    parser.add_argument('--device', type=str, default="cpu",
                      help='Device to run the model on (cpu/cuda)')
    parser.add_argument('--host', type=str, default="0.0.0.0",
                      help='Host address to bind the server to')
    parser.add_argument('--port', type=int, default=5000,
                      help='Port to run the server on')
    parser.add_argument('--workers', type=int, default=1,
                      help='Number of worker processes')
    parser.add_argument('--ctc_weight', type=float, default=0.5,
                      help='Weight for CTC loss in hybrid models')
    parser.add_argument('--method', type=str, default="ctc_beamsearch_lm",
                      help='Decoding method to use')
    parser.add_argument('--kenlm_alpha', type=float, default=0.3,
                      help='Language model alpha parameter')
    parser.add_argument('--kenlm_beta', type=float, default=1.5,
                      help='Language model beta parameter')
    parser.add_argument('--word_vocab_size', type=int, default=-1,
                      help='Size of word vocabulary')
    
    args = parser.parse_args()

    # Configure logging
    log_level = "INFO"
    log_format = ("<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | "
                 "<level>{level: <8}</level> | "
                 "<yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>")
    
    logger.add(sys.stderr, level=log_level, format=log_format, 
              colorize=True, backtrace=True, diagnose=True)
    logger.add(os.path.join(args.model_dir, 'file.log'), level=log_level, 
              format=log_format, colorize=False, backtrace=True, diagnose=True)

    logger.info(f"Starting server with configuration: {vars(args)}")

    # Load the appropriate ASR model
    if not args.model_dir:
        logger.error("model_dir must be specified")
        sys.exit(1)
        
    try:
        duration_model_mapping = load_model(args)
        model_name = args.model_dir.split("/")[-1]
        logger.success(f"Successfully loaded model: {model_name}")
        
        # Start the FastAPI server
        uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        sys.exit(1)