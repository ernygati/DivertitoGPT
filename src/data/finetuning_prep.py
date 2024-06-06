import pandas as pd
import json

class FinetuningTemplateMaker():
    def __init__(self,xlsx_filepath, save_filepath, llm_type = "yandex", num_first_rows = 51 ):
        self.xlsx_filepath = xlsx_filepath
        self.save_filepath =  save_filepath
        self.num_first_rows = num_first_rows
        self.llm_type = llm_type
    def _create_yandex_finetune_template(self, request_text, response_text):
        return {
        "request": request_text,
        "response": response_text
        }

    def _add_examples_in_json(self, json_file,request, response):
        if self.llm_type == "yandex":
            example = self._create_yandex_finetune_template(request, response)
        else:
            raise NotImplementedError
        json.dump(example, json_file, ensure_ascii=False)
        json_file.write(",\n")

    def make_template(self):
        finetune_df = pd.read_excel(self.xlsx_filepath).iloc[:self.num_first_rows][["query", "response"]]
        finetune_df

        with open(self.save_filepath, "w", encoding="utf-8") as f:
            for i in range(len(finetune_df)):
                request, response = finetune_df.iloc[i]
                self._add_examples_in_json(f, request, response)
        
    #check if json suits finetuning requirements
    # counter = 0
    # for i in range(len(finetune_df)):
    #     request, response = finetune_df.iloc[i]
    #     if (len(response)) >= 2000:
    #         counter+=1
    #         print("\n RESPONSE номер",i+2, len(response),response)