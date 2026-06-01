import asyncio
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket
from faster_whisper import WhisperModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

app = FastAPI()

# --- PUT YOUR HUGGING FACE TOKEN HERE ---
HF_TOKEN = "hf_your_actual_token_here"
# ----------------------------------------

# 1. Initialize Models (Load into VRAM once at startup)
print("Loading Models into VRAM...")

# Load ASR
asr_model = WhisperModel("base.en", device="cuda", compute_type="float16")

# Load LLM (Replace Qwen with Gemma and use the token)
tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-2b-it", 
    token=HF_TOKEN
)
llm_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it", 
    torch_dtype="auto", 
    device_map="cuda",
    token=HF_TOKEN
)

# 1. Initialize Models (Load into VRAM once at startup)
print("Loading Models into VRAM...")
asr_model = WhisperModel("base.en", device="cuda", compute_type="float16")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
llm_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", 
    torch_dtype="auto", 
    device_map="cuda"
)

def synthesize_audio(text):
    """
    Plug in your open TTS engine here (e.g., Voxtral, Kokoro, MeloTTS).
    Should accept text and return raw PCM audio bytes.
    """
    # ... TTS inference logic ...
    return b"simulated_audio_bytes"

system_prompt = "You are a friendly internal IT Helpdesk assistant. Keep answers brief and conversational."
chat_history = [{"role": "system", "content": system_prompt}]

@app.websocket("/ws/voice")
async def voice_agent_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_buffer = bytearray()
    
    try:
        while True:
            # 1. Receive Audio Stream from Client
            message = await websocket.receive_bytes()
            audio_buffer.extend(message)
            
            # 2. VAD & Chunking (Process roughly 1 second of audio at a time)
            if len(audio_buffer) > 16000 * 2:  # Assuming 16kHz, 16-bit
                audio_np = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                audio_buffer.clear()
                
                # ASR Stage
                segments, _ = asr_model.transcribe(audio_np, beam_size=1)
                user_text = "".join([segment.text for segment in segments]).strip()
                
                if user_text:
                    print(f"User: {user_text}")
                    chat_history.append({"role": "user", "content": user_text})
                    
                    # LLM Stage - Streaming setup
                    inputs = tokenizer.apply_chat_template(chat_history, return_tensors="pt", add_generation_prompt=True).to("cuda")
                    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                    generation_kwargs = dict(input_ids=inputs, streamer=streamer, max_new_tokens=150)
                    
                    # Run LLM generation in a separate thread so we can yield tokens
                    thread = Thread(target=llm_model.generate, kwargs=generation_kwargs)
                    thread.start()
                    
                    sentence_buffer = ""
                    assistant_full_reply = ""
                    
                    # TTS Stage - Sentence Boundary Detection
                    for new_token in streamer:
                        sentence_buffer += new_token
                        assistant_full_reply += new_token
                        
                        # Trigger TTS when a sentence is complete
                        if any(punct in sentence_buffer for punct in [".", "?", "!"]):
                            audio_chunk = synthesize_audio(sentence_buffer.strip())
                            
                            # Stream audio back to client immediately
                            await websocket.send_bytes(audio_chunk)
                            sentence_buffer = "" 
                    
                    # Append the final full reply to memory
                    chat_history.append({"role": "assistant", "content": assistant_full_reply})
                    
    except Exception as e:
        print(f"Connection closed: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)