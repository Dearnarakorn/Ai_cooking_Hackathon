import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from mockaiservice import setting
from mockaiservice.speech import tts
import gradio as gr

device = 0 if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    task="automatic-speech-recognition",
    model="biodatlab/whisper-th-medium-combined",
    chunk_length_s=30,
    device=device,
)

def ASR(input_path):
    # Perform ASR with the created pipe.
    lang = "en"
    result = pipe(input_path, generate_kwargs={"language": lang, "task": "transcribe"}, batch_size=16)["text"]
    return result

model_id = "scb10x/llama-3-typhoon-v1.5x-8b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

def LLM(input_msg):
    messages = [
        {"role": "system", "content": "You are a helpful assistant who always speaks English."},
        {"role": "user", "content": f"{input_msg}"},
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.4,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    result = tokenizer.decode(response, skip_special_tokens=True)
    return result

setting.set_api_key('ejjItkAMCvhD4Hr2U39B6INZt6nO5mlh')

def TTS(input_msg):
    tts.convert(input_msg, './output.wav')
    return "output.wav"

def transcribe_and_speak(audio):
    try:
        transcription = ASR(audio)  # Convert speech to text
    except Exception as e:
        print(f"Error: {e}")
        return "Error in processing ASR", None
    
    try:
        llm_output = LLM(transcription)  # Get response from language model
    except Exception as e:
        print(f"Error: {e}")
        return "Error in processing LLM", None


    try:
        tts_output = TTS(llm_output)  # Convert response to speech
        return transcription, tts_output
    except Exception as e:
        print(f"Error: {e}")
        return "Error in processing TTS", None
    

    
interface = gr.Interface(
    fn=transcribe_and_speak,
    inputs=gr.Audio(type="filepath"),
    outputs=["text", "audio"],
)

interface.launch(server_name="0.0.0.0", server_port=8000, debug=True)
