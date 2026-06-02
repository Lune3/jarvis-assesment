import asyncio
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket
from faster_whisper import WhisperModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# Import Kokoro TTS
from kokoro import KPipeline

app = FastAPI()

# =====================================================================
# 🛑 PASTE YOUR HUGGING FACE TOKEN HERE
# =====================================================================
HF_TOKEN = "token"

print("Loading Models into VRAM. This will take a moment...")

# 1. Load ASR (Speech-to-Text)
asr_model = WhisperModel("base.en", device="cuda", compute_type="float16")

# 2. Load LLM (Reasoning)
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

# 3. Load Kokoro TTS Pipeline ('a' stands for American English)
tts_pipeline = KPipeline(lang_code='a')

print("Models loaded successfully!")

# =====================================================================
# KOKORO AUDIO SYNTHESIS
# =====================================================================
def synthesize_audio(text: str) -> bytes:
    """
    Takes a string of text, generates speech using Kokoro, 
    and returns raw 16-bit PCM audio bytes for the WebSocket.
    """
    print(f"[TTS Generating]: {text}")
    
    # 'af_heart' is a highly natural-sounding female American voice
    generator = tts_pipeline(text, voice='af_heart', speed=1.0)
    
    all_audio = []
    for _, _, audio_numpy in generator:
        all_audio.append(audio_numpy)
        
    if not all_audio:
        return b""
        
    # Combine chunks (if any)
    combined_audio = np.concatenate(all_audio)
    
    # Kokoro outputs float32 audio natively. 
    # Convert it to standard 16-bit PCM for WebSocket transmission.
    audio_int16 = (combined_audio * 32767).astype(np.int16)
    
    return audio_int16.tobytes()

# =====================================================================
# WEBSOCKET LOGIC
# =====================================================================
system_prompt = "You are a friendly internal IT Helpdesk assistant. Keep answers brief and conversational."
chat_history = [{"role": "system", "content": system_prompt}]

@app.websocket("/ws/voice")
async def voice_agent_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_buffer = bytearray()
    print("Client connected via WebSocket.")
    
    try:
        while True:
            # Receive Audio Stream from Client
            message = await websocket.receive_bytes()
            audio_buffer.extend(message)
            
            # Process roughly 1 second of audio at a time (assuming 16kHz, 16-bit mono input)
            if len(audio_buffer) > 16000 * 2:  
                audio_np = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                audio_buffer.clear()
                
                # --- ASR STAGE ---
                segments, _ = asr_model.transcribe(audio_np, beam_size=1)
                user_text = "".join([segment.text for segment in segments]).strip()
                
                if user_text:
                    print(f"\nUser said: {user_text}")
                    chat_history.append({"role": "user", "content": user_text})
                    
                    # --- LLM STAGE ---
                    inputs = tokenizer.apply_chat_template(
                        chat_history, 
                        return_tensors="pt", 
                        add_generation_prompt=True
                    ).to("cuda")
                    
                    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                    generation_kwargs = dict(input_ids=inputs, streamer=streamer, max_new_tokens=150)
                    
                    thread = Thread(target=llm_model.generate, kwargs=generation_kwargs)
                    thread.start()
                    
                    sentence_buffer = ""
                    assistant_full_reply = ""
                    
                    # --- TTS & STREAMING STAGE ---
                    for new_token in streamer:
                        sentence_buffer += new_token
                        assistant_full_reply += new_token
                        
                        # Trigger TTS when a full sentence is formed
                        if any(punct in sentence_buffer for punct in [".", "?", "!"]):
                            # Generate and send audio immediately
                            audio_chunk = synthesize_audio(sentence_buffer.strip())
                            await websocket.send_bytes(audio_chunk)
                            sentence_buffer = "" 
                    
                    chat_history.append({"role": "assistant", "content": assistant_full_reply})
                    print(f"Assistant replied: {assistant_full_reply}")
                    
    except Exception as e:
        print(f"WebSocket connection closed: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
