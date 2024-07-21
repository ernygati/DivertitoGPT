import os
import pandas as pd
import json
from src.rag_chain import SYSTEM_PROMPT

global system_content
system_content = " ".join(SYSTEM_PROMPT.split("\n")[:-7])

global num_characters 
num_characters_list= []

class FinetuningTemplateMaker():
    def __init__(self,xlsx_filepath, save_path, llm_type = "yandex", num_first_rows = 51,val_len = 2 ):
        self.xlsx_filepath = xlsx_filepath
        self.save_path =  save_path
        self.num_first_rows = num_first_rows
        self.llm_type = llm_type
        self.val_len = val_len
    def _create_yandex_finetune_template(self, request_text, response_text):
        return {
        "request": request_text,
        "response": response_text
        }
        
    def _create_openai_finetune_template(self,system_content, user_content, assistant_content):
        num_characters_list.append(len(system_content)+len(user_content) + len(assistant_content))
        return {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        }

    def _add_examples_in_json(self, json_file,request, response):
        if self.llm_type == "yandex":
            example = self._create_yandex_finetune_template(request, response)
        elif self.llm_type == "openai":
            example = self._create_openai_finetune_template(system_content, request,response)
            
        json.dump(example, json_file, ensure_ascii=False)
        json_file.write(",\n")

    def make_template(self):
        finetune_df = pd.read_excel(self.xlsx_filepath, usecols= ["query", "response"],
                                    nrows=self.num_first_rows)
        if self.llm_type == "yandex":
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            with open(self.save_path, "w", encoding="utf-8") as f:
                for i in range(len(finetune_df)):
                    request, response = finetune_df.iloc[i]
                    self._add_examples_in_json(f, request, response)
        elif self.llm_type == "openai":
            finetune_df = finetune_df.sample(frac=1)
            train_list = finetune_df.iloc[:len(finetune_df)-self.val_len,:].values
            val_list = finetune_df.iloc[len(finetune_df)-self.val_len:,:].values

            os.makedirs(self.save_path, exist_ok=True)
            with open(os.path.join(self.save_path, "finetune_responces_train.jsonl"), 
                    "w", encoding="utf-8") as train_f:
                for (tq, tr) in train_list:
                        self._add_examples_in_json(train_f, tq, tr)
                        
            with open(os.path.join(self.save_path, "finetune_responces_val.jsonl"),
                    "w", encoding="utf-8") as val_f:
                for (vt, vr) in val_list:
                        self._add_examples_in_json(val_f, vt, vr)
        
    #check if json suits finetuning requirements
    # counter = 0
    # for i in range(len(finetune_df)):
    #     request, response = finetune_df.iloc[i]
    #     if (len(response)) >= 2000:
    #         counter+=1
    #         print("\n RESPONSE номер",i+2, len(response),response)