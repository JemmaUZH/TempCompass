# -*- coding: utf-8 -*-
import os, json, argparse, base64, time, cv2
from tqdm import tqdm
from volcenginesdkarkruntime import Ark

client = Ark(
    base_url="...",  
    api_key=os.environ.get("ARK_API_KEY"),  
)

# 均匀抽取帧并转为 image_url 格式（base64）
def extract_frames_imageurls(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return []
    frame_indices = [int(i * total_frames / num_frames + total_frames / (2 * num_frames)) for i in range(num_frames)]

    image_urls = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        img_url = f"data:image/jpeg;base64,{img_base64}" 
        image_urls.append(img_url)
    cap.release()
    return image_urls


def get_response_with_imageurls(image_urls, prompt, model="ep-20250703140221-4kncm"):
    content = [{"type": "image_url", "image_url": {"url": img}} for img in image_urls]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            extra_headers={'x-is-encrypted': 'true'} 
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[Ark Error]: {e}")
        return ""


def inference_single_video(video_path, prompt, maxtry=10):
    while maxtry > 0:
        try:
            frames = extract_frames_imageurls(video_path, num_frames=8)
            response = get_response_with_imageurls(frames, prompt)
            time.sleep(1)
            return response
        except Exception as e:
            print(f"[Retry] {maxtry} left... {e}")
            maxtry -= 1
            time.sleep(5)
    return ""


answer_prompt = {
    "multi-choice": "\nPlease directly give the best option:",
    "yes_no": "\nPlease answer yes or no:",
    "caption_matching": "\nPlease directly give the best option:",
    "captioning": ""
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='path_to_tempcompass')
    parser.add_argument('--output_path', default='predictions/ark')
    parser.add_argument('--task_type', default='multi-choice', choices=['multi-choice', 'captioning', 'caption_matching', 'yes_no'])
    args = parser.parse_args()

    question_path = os.path.join(args.data_path, 'questions', f"{args.task_type}.json")
    with open(question_path, 'r') as f:
        input_datas = json.load(f)

    os.makedirs(args.output_path, exist_ok=True)
    pred_file = os.path.join(args.output_path, f"{args.task_type}.json")
    if os.path.isfile(pred_file):
        with open(pred_file, 'r') as f:
            predictions = json.load(f)
    else:
        predictions = {}

    for vid, data in tqdm(input_datas.items()):
        if vid in predictions:
            continue
        predictions[vid] = {}
        video_path = os.path.join(args.data_path, 'videos', f'{vid}.mp4')
        for dim, questions in data.items():
            predictions[vid][dim] = []
            for question in questions:
                inp = question['question'] + answer_prompt[args.task_type]
                ark_response = inference_single_video(video_path, inp)
                predictions[vid][dim].append({
                    'question': question['question'],
                    'answer': question['answer'],
                    'prediction': ark_response
                })
        with open(pred_file, 'w') as f:
            json.dump(predictions, f, indent=4)

    print(f"✅ Ark 多模态推理完成，结果已写入：{pred_file}")
