import os
import jwt
import requests
import time
from yandex_chain import YandexLLM
from dotenv import load_dotenv
from typing import Any, List, Mapping, Optional
import langchain_core
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_openai import ChatOpenAI



load_dotenv()
YANDEX_SERVICE_ACCOUNT_ID = os.environ.get("YANDEX_SERVICE_ACCOUNT_ID")
YANDEX_KEY_ID = os.environ.get("YANDEX_KEY_ID")
YANDEX_FOLDER_ID = os.environ.get("YANDEX_FOLDER_ID")
YANDEX_API_KEY = os.environ.get("YANDEX_API_KEY")

print(os.getcwd())
with open("../yandex_private_key.txt") as file:
    YANDEX_PRIVATE_KEY = file.read()

class ChatLLM():
    def __init__(self):
        pass
    
    def _generate_yandex_token(self):

        # Получаем IAM-токен 
        now = int(time.time()) 
        payload = { 
                'aud': 'https://iam.api.cloud.yandex.net/iam/v1/tokens', 
                'iss': YANDEX_SERVICE_ACCOUNT_ID, 
                'iat': now, 
                'exp': now + 360} 
        
        # Формирование JWT 
        encoded_token = jwt.encode( 
            payload, 
            YANDEX_PRIVATE_KEY, 
            algorithm='PS256', 
            headers={'kid': YANDEX_KEY_ID}) 
        
        url = 'https://iam.api.cloud.yandex.net/iam/v1/tokens' 
        x = requests.post(url,   
                        headers={'Content-Type': 'application/json'}, 
                        json = {'jwt': encoded_token}).json() 
        token = x['iamToken']
        
        return token
        
    def get_yandex_llm(self, base = True):
        if base:
            llm = YandexLLM(folder_id=YANDEX_FOLDER_ID, api_key=YANDEX_API_KEY)
        else:
            global token
            token = self._generate_yandex_token()
            llm = YandexDiveritoLLM(iam_token=token,folder_id = YANDEX_FOLDER_ID, 
                                       use_lite=True,temperature = 0.2, max_tokens = 1000)
        return llm
    
    def get_opeanai_llm(self):
        llm = ChatOpenAI( temperature=0.1)
        return llm

            

class YandexDiveritoLLM(langchain_core.language_models.llms.LLM):
    api_key: str = None
    iam_token: str = None
    folder_id: str = None
    max_tokens : int = 1500
    temperature : float = 1
    prompt : str = None

    @property
    def _llm_type(self) -> str:
        return "yagpt"

    def _call(
        self,
        question: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        catalog_id = "b1gti4ai7mdnk1aocko1"
        model_id = "bt1oknfh0miqf32glf2a"
        req = {
                "modelUri": "ds://bt1oknfh0miqf32glf2a",
                "completionOptions": {
                    "stream": False,
                    "temperature": 0.6,
                    "maxTokens": 7500
                },
                "messages": [
                    {
                    "role": "system",
                    "text": "Представь, что ты Максим Дивертито - известный блоггер и коуч. Ты знаешь все в области пикапа, соблазнения и построения отношений. Ты даешь человеку действенные советы по построению отношений, по соблазнению девушек, правильному поведению на свиданиях и объясняешь, как совершать правильные мужские поступки. Ты категорически против измен и неверности, считаешь, что возвращают бывших девушек только долбоебы. Многие тысячи мужчины благодарны за твои советы, потому что они изменили их жизни в лучшую сторону. Ты можешь использовать мат и нецензурную лексику в своих ответах, отвечать саркастично, шутливо или даже грубо."
                    },
                    {
                    "role": "user",
                    "text": question
                    }
                ]    
        }
        headers = {"Authorization" : "Bearer " + token,
                           "x-folder-id": catalog_id, }
        res = requests.post("https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
            headers=headers, json=req).json()
        return res['result']['alternatives'][0]['message']["text"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"max_tokens": self.max_tokens, "temperature" : self.temperature }
