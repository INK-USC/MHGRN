from transformers import (OpenAIGPTConfig, OpenAIGPTModel, OpenAIGPTTokenizer,
                          BertConfig, BertModel, BertTokenizer,
                          XLNetConfig, XLNetModel, XLNetTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

for model_type, model_name, config_class, model_class, tokenizer_class in (
        ('gpt', 'openai-gpt', OpenAIGPTConfig, OpenAIGPTModel, OpenAIGPTTokenizer),
        ('bert', 'bert-base-uncased', BertConfig, BertModel, BertTokenizer),
        ('bert', 'bert-large-uncased', BertConfig, BertModel, BertTokenizer),
        ('xlnet', 'xlnet-large-cased', XLNetConfig, XLNetModel, XLNetTokenizer),
        ('roberta', 'roberta-large', RobertaConfig, RobertaModel, RobertaTokenizer),
):
    config = config_class.from_pretrained(model_name)
    tokenizer = tokenizer_class.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name)
