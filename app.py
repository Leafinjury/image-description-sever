from flask import Flask, request
import asyncio
import json
from urllib.parse import urljoin
import threading
from aiohttp import ClientSession
import traceback
import base64
# Add this import at the top of your file
from concurrent.futures import ThreadPoolExecutor


from config import *

app = Flask(__name__)

# 处理单个图片的异步函数
async def process_single_image(image, session):
    image_url = urljoin(DOCLING_SERVER_URL, image)
    
    # 要发送的JSON数据
    model_data = {
        "model": "Qwen2.5-VL-3B-Instruct-AWQ",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "用理性的方式描述这张图片"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ],
        "max_tokens": 600
    }

    # 设置请求头部
    model_headers = {
        "Content-Type": "application/json",
    }

    # 发送异步POST请求
    model_url = MODEL_URL + "v1/chat/completions"
    
    async with session.post(model_url, json=model_data, headers=model_headers) as response:
        # 检查请求是否成功
        if response.status == 200:
            try:
                # 获取响应文本
                response_text = await response.text()
                
                # 将字符串转换为JSON对象
                response_json = json.loads(response_text)
                sentence = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                return image, sentence
            
            except json.JSONDecodeError as e:
                print("返回的数据不是有效的JSON格式:", e)
                return image, ""
        else:
            print(f"请求模型失败，状态码：{response.status}，响应内容：{await response.text()}")
            return image, ""

# 发送回调请求的异步函数
async def send_callback_request(images_description, doc_id, session, success=True, error_msg=None):
    # 向图片描述回调接口发送POST请求
    description_data = {
        "code": 0 if success else 1,
        "message": "success" if success else error_msg,
        "data": images_description,
        "doc_id": doc_id
    }
    print(f"回调数据: {description_data}")
    # 设置请求头部
    description_headers = {
        "Content-Type": "application/json",
    }
    description_url = urljoin(DOCLING_SERVER_URL, "picture_result")
    
    try:
        async with session.post(description_url, json=description_data, headers=description_headers) as response:
            if response.status == 200:
                print("回调接口请求成功")
            else:
                print(f"回调接口请求失败，状态码：{response.status}")
    except Exception as e:
        print(f"发送回调请求时发生异常：{str(e)}")

# 处理所有图片的异步主函数
async def process_images_task(images, doc_id):
    images_description = {}
    
    try:
        # 创建一个HTTP会话，用于所有请求
        async with ClientSession() as session:
            try:
                # 为每个图片创建一个异步任务
                tasks = [process_single_image(image, session) for image in images]
                
                # 并发执行所有任务
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 处理结果
                for result in results:
                    if isinstance(result, Exception):
                        print(f"处理图片时发生错误: {result}")
                    else:
                        image, description = result
                        if description:  # 只有当描述不为空时才添加
                            images_description[image] = description
                
                # 所有图片处理完成后，发送成功回调请求
                await send_callback_request(images_description, doc_id, session)
                
            except Exception as e:
                # 捕获处理过程中的任何异常
                error_msg = f"处理图片任务时发生异常: {str(e)}"
                print(error_msg)
                # 发送错误回调请求
                await send_callback_request(images_description, doc_id, session, False, error_msg)
    except Exception as e:
        # 如果连ClientSession创建都失败，则需要创建一个新的会话来发送错误信息
        error_msg = f"创建会话失败: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        try:
            async with ClientSession() as error_session:
                await send_callback_request({}, doc_id, error_session, False, error_msg)
        except Exception as session_error:
            print(f"创建错误回调会话也失败了: {str(session_error)}")

# 在单独线程中运行异步任务的辅助函数
def run_async_task(coroutine):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coroutine)
    except Exception as e:
        print(f"异步任务执行失败: {str(e)}\n{traceback.format_exc()}")
    finally:
        loop.close()

@app.route('/image_description', methods=['POST'])
def generateImageIescription():
    # 获取POST请求发送的JSON数据
    request_data = request.get_json()
    
    images = request_data.get("list", [])
    doc_id = request_data.get("doc_id", None)
    
    # 在单独的线程中启动异步任务
    thread = threading.Thread(
        target=run_async_task, 
        args=(process_images_task(images, doc_id),)
    )
    thread.daemon = True
    thread.start()
    
    return {
        "code": 0,
        "message": "success"
    }

@app.route('/image_file_description', methods=['POST'])
def generateImageFileIescription():
    # 获取上传的文件
    image_file = request.files.get("image_file")
    
    # 获取文件内容类型
    content_type = image_file.content_type
    
    # 检查文件类型是否支持
    supported_types = {
        "image/png": "png",
        "image/jpeg": "jpeg", 
        "image/jpg": "jpeg",  # 处理jpg格式
        "image/webp": "webp"
    }
    
    if content_type not in supported_types:
        return {
            "status_code": 400,
            "content": {
                "error": f"不支持的文件类型: {content_type}。仅支持PNG、JPEG和WEBP格式。"
            }
        }
    
    # 读取文件内容
    file_content = image_file.read()
    base64_image = base64.b64encode(file_content).decode("utf-8")
    
    # 使用正确的MIME类型
    image_format = supported_types[content_type]
    
    # 将异步操作封装到一个异步函数中
    async def process_image():
        # 要发送的JSON数据
        model_data = {
            "model": "Qwen2.5-VL-3B-Instruct-AWQ",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "用理性的方式描述这张图片"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{image_format};base64,{base64_image}"}, 
                        }
                    ]
                }
            ],
            "max_tokens": 600
        }

        # 设置请求头部
        model_headers = {
            "Content-Type": "application/json",
        }

        # 发送异步POST请求
        model_url = MODEL_URL + "v1/chat/completions"
        
        # 创建一个HTTP会话，用于所有请求
        async with ClientSession() as session:
            async with session.post(model_url, json=model_data, headers=model_headers) as response:
                # 检查请求是否成功
                if response.status == 200:
                    try:
                        # 获取响应文本
                        response_text = await response.text()

                        # 将字符串转换为JSON对象
                        response_json = json.loads(response_text)
                        sentence = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                        
                        # 返回成功状态码和句子内容
                        return {
                            "status_code": 200,
                            "content": {"description": sentence}
                        }
                        
                    except json.JSONDecodeError as e:
                        print("返回的数据不是有效的JSON格式:", e)
                        return {
                            "status_code": 500,
                            "content": {"error": f"解析模型响应时出错, 返回的数据不是有效的JSON格式: {str(e)}"}
                        }
                else:
                    print(f"请求模型失败，状态码：{response.status}，响应内容：{await response.text()}")
                    return {
                        "status_code": response.status,
                        "content": {"error": f"模型请求失败，状态码：{response.status}，响应内容：{await response.text()}"}
                    }
    
    # 在一个线程中运行异步函数
    def run_async_task():
        return asyncio.run(process_image())
    
    # 创建线程，运行异步任务，并等待结果
    with ThreadPoolExecutor() as executor:
        future = executor.submit(run_async_task)
        result = future.result()  # 这里会阻塞直到异步操作完成
    
    return result


if __name__ == '__main__':
    app.run(
        debug=FLASK_CONFIG['DEBUG'], 
        host=FLASK_CONFIG['HOST'], 
        port=FLASK_CONFIG['PORT']
    )
