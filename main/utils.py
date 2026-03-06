import torch
import os
import pickle
import numpy as np
from transformers import AutoTokenizer
import copy
from datasets import load_dataset

# convert, but just return text.
def create_prompt_with_tulu_chat_format(messages, tokenizer, add_bos=True, prompt_only=False, response_only=False, no_special_tokens=False):
    formatted_text = ""
    if response_only:
        if messages[0]['role'] == 'system':
            messages = [messages[2]]
        else:
            messages = [messages[1]]
    elif prompt_only:
        if messages[0]['role'] == 'system':
            messages = messages[:2]
        else:
            messages = messages[:1]
    for message in messages:
        if message["role"] == "system":
            formatted_text += "<|system|>\n" + message["content"] + "\n"
        elif message["role"] == "user":
            formatted_text += "<|user|>\n" + message["content"] + "\n"
        elif message["role"] == "assistant":
            formatted_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
        else:
            raise ValueError(
                "Tulu chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(
                    message["role"]
                )
            )
    formatted_text += "<|assistant|>\n"
    if response_only or prompt_only:
        formatted_text = formatted_text.replace("<|assistant|>\n", "").replace(tokenizer.eos_token, "")
    if no_special_tokens:
        if tokenizer.bos_token:
            formatted_text = formatted_text.replace(tokenizer.bos_token, "")
        if tokenizer.eos_token:
            formatted_text = formatted_text.replace(tokenizer.eos_token, "")
    return formatted_text

def encode_prompts(dataset, example, tokenizer, max_seq_length, include_response=True, response_only=False, only_first_two=False, prompt_only=False, add_bos_token=False):
    if "ling" in dataset:
        return encode_with_messages_format_ling(example, tokenizer, max_seq_length)
    elif "tulu" in dataset:
        return encode_with_messages_format_tulu(example, tokenizer, max_seq_length)
    else:
        return encode_with_messages_format(example, tokenizer, max_seq_length, include_response=True, response_only=False, only_first_two=False, prompt_only=False, add_bos_token=False)

def encode_with_messages_format_ling(example, tokenizer, max_seq_length):
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")
    message_text = ""
    if "system_message" in example.keys() and example["system_message"] is not None:
        message_text += "<|system|>\n" + example["system_message"].strip() + "\n"
    for message in messages:
        if message["role"] == "HUMAN":
            message_text += "<|user|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "OBSERVATION":
            message_text += "<|observation|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "ASSISTANT":
            message_text += "<|assistant|>\n" + message["content"].strip() + "\n"
        else:
            raise ValueError("Invalid role: {}".format(message["role"]))
    message_text = message_text.strip()
    message_text = message_text + tokenizer.eos_token
    # print(message_text)
    tokenized_example = tokenizer(message_text, return_tensors="pt", max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    attention_mask = tokenized_example.attention_mask

    return {
        "input_ids": input_ids.flatten(),
        "attention_mask": attention_mask.flatten(),
    }

def encode_with_messages_format_tulu(example, tokenizer, max_seq_length):
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")
    message_text = ""
    if "system" in example.keys() and example["system"] is not None:
        message_text += "<|system|>\n" + example["system"].strip() + "\n"
    for message in messages:
        if message["role"] == "user":
            message_text += "<|user|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "assistant":
            message_text += "<|assistant|>\n" + message["content"].strip() + "\n"
        elif message["role"] != "system":
            raise ValueError("Invalid role: {}".format(message["role"]))
    message_text = message_text.strip()
    message_text = message_text + tokenizer.eos_token
    # print(message_text)
    tokenized_example = tokenizer(message_text, return_tensors="pt", max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    attention_mask = tokenized_example.attention_mask

    return {
        "input_ids": input_ids.flatten(),
        "attention_mask": attention_mask.flatten(),
    }

# needed for open-instruct: convert msg format.
def encode_with_messages_format(example, tokenizer, max_seq_length, include_response=True, response_only=False, only_first_two=False, prompt_only=False, add_bos_token=False):
    """
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    """
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")

    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        # add bos token if needed
        if add_bos_token:
            message_text = tokenizer.bos_token + message_text
        return message_text

    def _concat_messages_ling(example, msg_len=-1):
        if msg_len < 0:
            if isinstance(example, list):
                print(example)
            messages = example["messages"]
        else:
            messages = example["messages"][:msg_len]
        message_text = ""
        if "system_message" in example.keys() and example["system_message"] is not None:
            message_text += "<|system|>\n" + example["system_message"].strip() + "\n"
        for message in messages:
            if message["role"] == "HUMAN":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "OBSERVATION":
                message_text += "<|observation|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "ASSISTANT":
                message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        # add bos token if needed
        if add_bos_token:
            message_text = tokenizer.bos_token + message_text
        return message_text

    # change: just take the first two prompts.
    if only_first_two:
        # if first role is system, we actually want to take the second and third message,
        # ignoring the first system message.
        if messages[0]["role"] == "system":
            messages = messages[1:3]
        else:
            messages = messages[:2]
    if prompt_only:
        if messages[0]["role"] == "system":
            messages = messages[:2]
        else:
            messages = messages[:1]
        msg = _concat_messages(messages).strip() + tokenizer.eos_token  # add eos token manually.
        res = tokenizer(msg, return_tensors="pt", max_length=max_seq_length, truncation=True)
        return {
            "string": msg,
            "input_ids": res.input_ids.flatten(),
            "attention_mask": res.attention_mask.flatten(),
            'labels': res.input_ids.flatten()
        }
    elif response_only:
        idx = 1
        if messages[0]["role"] == "system":
            idx = 2
        msg = "<|assistant|>\n" + messages[idx]["content"].strip() + tokenizer.eos_token
        res = tokenizer(msg, return_tensors="pt", max_length=max_seq_length, truncation=True)
        return {
            "string": msg,
            "input_ids": res.input_ids.flatten(),
            "attention_mask": res.attention_mask.flatten(),
            'labels': res.input_ids.flatten()
        }

    if "mid" in example.keys():
        example_text = _concat_messages_ling(example).strip() 
    else:
        example_text = _concat_messages(messages).strip()
    # print(example_text)
    tokenized_example = tokenizer(example_text, return_tensors="pt", max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    if "mid" in example.keys():
        messages = example["messages"]
        overall_start_idx = 0
        if "system_message" in example.keys() and example["system_message"] is not None:
            messages_so_far = "<|system|>\n" + example["system_message"].strip() + "\n"
            messages_end_idx = tokenizer(
                    messages_so_far, return_tensors="pt", max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
            labels[:, :messages_end_idx] = -100
            overall_start_idx = messages_end_idx
            # print(labels)
        for message_idx, message in enumerate(messages):
            if message["role"] != "ASSISTANT":
                if message_idx == 0:
                    assert overall_start_idx >= 0
                    message_start_idx = overall_start_idx
                else:
                    copied_ex = copy.deepcopy(example)
                    example["messages"] = messages[:message_idx]
                    message_start_idx = tokenizer(
                            _concat_messages_ling(copied_ex),
                            return_tensors="pt",
                            max_length=max_seq_length,
                            truncation=True,
                        ).input_ids.shape[1]
                if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "ASSISTANT":
                    # here we also ignore the role of the assistant
                    messages_so_far = _concat_messages_ling(example, message_idx + 1) + "<|assistant|>\n"
                else:
                    messages_so_far = _concat_messages_ling(example, message_idx + 1)
                message_end_idx = tokenizer(
                    messages_so_far, return_tensors="pt", max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
                labels[:, message_start_idx:message_end_idx] = -100

                if message_end_idx >= max_seq_length:
                    break
                # print(labels)
    else:
        for message_idx, message in enumerate(messages):
            if message["role"] != "assistant":
                if message_idx == 0:
                    message_start_idx = 0
                else:
                    message_start_idx = tokenizer(
                        _concat_messages(messages[:message_idx]),
                        return_tensors="pt",
                        max_length=max_seq_length,
                        truncation=True,
                    ).input_ids.shape[1]
                if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                    # here we also ignore the role of the assistant
                    messages_so_far = _concat_messages(messages[: message_idx + 1]) + "<|assistant|>\n"
                else:
                    messages_so_far = _concat_messages(messages[: message_idx + 1])
                message_end_idx = tokenizer(
                    messages_so_far, return_tensors="pt", max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
                labels[:, message_start_idx:message_end_idx] = -100

                if message_end_idx >= max_seq_length:
                    break

    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
        "string": messages_so_far,
    }

def get_appropriate_data_dir():
    return "/"

class InMemoryFaiss:
    def __init__(self):
        self.is_trained = False
        self.vectors = []

    def add(self, vectors):
        self.vectors.extend(vectors)

    def search(self, query, k):
        # compute inner product between query and all vectors
        query = query.reshape(-1, 1)
        scores = np.stack(self.vectors, axis=0) @ query
        scores = scores[:, 0]
        # sort by score
        sorted_scores = np.argsort(scores)
        # return raw numbers and indices
        return scores[sorted_scores[:k]][None, ], sorted_scores[:k][None, ]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.vectors, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.vectors = pickle.load(f)
        self.is_trained = True

def get_train_dataset(root):
  if os.path.isdir(root):
    file_list = []
    for root, dirs, files in os.walk(root):
      for file in files:
        file_list.append(os.path.join(root, file))
    file_list = sorted(file_list)
    train_dataset = load_dataset('json', data_files=file_list, split='train')
  return train_dataset

def merge_by_segment(x: torch.Tensor) -> torch.Tensor:
    n, B, D = x.shape
    assert D % n == 0
    seg = D // n

    parts = []
    for i in range(n):
        start = i * seg
        end = (i + 1) * seg
        parts.append(x[i, :, start:end]) 

    out = torch.cat(parts, dim=-1) 
    return out

def get_batch_embeddings(inputs, 
    model, 
    pooling, 
    n_layer, 
    sec_pooling, 
    projector,
    take_layer=-1
):
    with torch.no_grad():
        outputs = model(
            **{k: v for k, v in inputs.items() if k != "labels"}, output_hidden_states=True
        )
    # label_len = torch.sum(train_inputs["labels"] != -100, dim=1)
    input_lens = torch.sum(inputs["attention_mask"], dim=1)
    # Get the mean hidden state corresponding to the label
    if pooling == "mean":
        embeddings = torch.mean(outputs["hidden_states"][take_layer], dim=1)
        embeddings = embeddings.unsqueeze(1)  # for compat
    elif pooling == "weighted_mean":
        hidden_states = outputs["hidden_states"][take_layer] # B, L, D
        weighting_mask = torch.arange(hidden_states.size(1), device=hidden_states.device).unsqueeze(0) + 1  # +1
        weighting_mask = weighting_mask / weighting_mask.sum(dim=1, keepdim=True) # 1, L
        embeddings = torch.sum(hidden_states * weighting_mask.unsqueeze(-1), dim=1) # B, D
        embeddings = embeddings.unsqueeze(1)  # for compat B, 1, D
    elif pooling == "multilayer":
        hidden_states = outputs["hidden_states"]
        total_layers = len(hidden_states) - 1 
        assert isinstance(n_layer, int) and total_layers % n_layer == 0 and n_layer > 0, f"n_layer must be a positive integer that divides {total_layers}"

        step = total_layers // n_layer
        selected_indices = [step * i for i in range(1, n_layer + 1)]
        selected_hidden_states = [hidden_states[i] for i in selected_indices]

        stacked_hidden_states = torch.stack(selected_hidden_states, dim=0)  # n, B, L, D
        weighting_mask = torch.arange(stacked_hidden_states.size(2), device=stacked_hidden_states.device) + 1  # +1
        weighting_mask = weighting_mask / weighting_mask.sum() # L
        embeddings = torch.sum(stacked_hidden_states * weighting_mask.unsqueeze(-1).unsqueeze(0).unsqueeze(0), dim=2) # n, B, D
        if sec_pooling == "sum":
            embeddings = torch.sum(embeddings, dim=0)
        elif sec_pooling == "cat": # Not maintained
            embeddings = merge_by_segment(embeddings)
        elif sec_pooling == "redcat":
            if n_layer == 1:
                embeddings = embeddings[0]  # [1, B, D] -> [B, D]
            else:
                red_tensor_list = []
                for idx, layer in enumerate(embeddings):
                    red_tensor = projector.project(layer, model_id=idx)
                    red_tensor_list.append(red_tensor)
                embeddings = torch.cat(red_tensor_list, dim=1)
        else:
            raise ValueError(f"Unsupported sec_pooling: {sec_pooling}")
        embeddings = embeddings.unsqueeze(1)
    elif pooling == "lastlayer":
        hidden_states = outputs["hidden_states"]
        assert isinstance(n_layer, int) and n_layer > 0, "n_layer must be a positive integer"
        selected_indices = range(-n_layer, 0)
        selected_hidden_states = [hidden_states[i] for i in selected_indices]

        stacked_hidden_states = torch.stack(selected_hidden_states, dim=0)  # n, B, L, D
        weighting_mask = torch.arange(stacked_hidden_states.size(2), device=stacked_hidden_states.device) + 1  # +1
        weighting_mask = weighting_mask / weighting_mask.sum() # L
        embeddings = torch.sum(stacked_hidden_states * weighting_mask.unsqueeze(-1).unsqueeze(0).unsqueeze(0), dim=2) # n, B, D
        if sec_pooling == "avg":
            embeddings = torch.mean(embeddings, dim=0)
        elif sec_pooling == "redcat":
            if n_layer == 1:
                embeddings = embeddings[0]  
            else:
                red_tensor_list = []
                for idx, layer in enumerate(embeddings):
                    red_tensor = projector.project(layer, model_id=idx)
                    red_tensor_list.append(red_tensor)
                embeddings = torch.cat(red_tensor_list, dim=1)
        else:
            raise ValueError(f"Unsupported sec_pooling: {sec_pooling}")
        embeddings = embeddings.unsqueeze(1)
    else:
        embeddings = outputs["hidden_states"][take_layer][:, input_lens - 1]
    return embeddings

if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/models/Qwen2.5-7B", use_fast=True)

    example = {"id":"oasst1_5100","messages":[{"content":"\u00bfCu\u00e1l consideraras que es un buen indicador econ\u00f3mico para medir el crecimiento econ\u00f3mico de un paisa en desarrollo?","role":"user"},{"content":"Es dificil elegir un unico indicador ya que distintos indicadores reflejan diferentes caracteristicas de un pais. Muchos de ellos pueden ser antagonistas de manera que mejorar uno puede empeorar el otro, de manera que para evaluar un pais hay que contemplar m\u00e1s de uno. Adem\u00e1s, as\u00ed como los diferentes indicadores se enfocan en distintas caracter\u00edsticas, distintos observadores pueden ponderar de manera diferente los aspectos relevantes de un crecimiento economico. Algunos de los mas usuales son el PBI o el PBI per c\u00e1pita","role":"assistant"}]}
    test1 = encode_prompts("tulu3", example, tokenizer, 2048)

    example = {"dataset": "flan_v2", "id": "flan_v2_5", "messages": [{"role": "system", "content":""}, {"role": "user", "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nGiven two lists of strings, output the length of the longest common substring.\n\n### Input:\nlist1 = [\"hello\", \"world\", \"happiness\"]\nlist2 = [\"howdy\", \"world\", \"happiness\"]"}, {"role": "assistant", "content": "To solve the problem of finding the **length of the longest common substring** between two lists of strings, we need to compare all possible substrings from each string in `list1` with all possible substrings from each string in `list2`, and find the maximum length of any common substrings.\n\n### Step-by-step approach:\n1. Iterate over all pairs of strings from `list1` and `list2`.\n2. For each pair of strings, find the **longest common substring**.\n3. Track the maximum length found across all pairs.\n\nWe'll use a dynamic programming approach to compute the longest common substring for each string pair.\n\n### Python-style pseudocode logic:\n```python\ndef longest_common_substring_length(str1, str2):\n    m = len(str1)\n    n = len(str2)\n    dp = [[0] * (n+1) for _ in range(m+1)]\n    max_len = 0\n    for i in range(1, m+1):\n        for j in range(1, n+1):\n            if str1[i-1] == str2[j-1]:\n                dp[i][j] = dp[i-1][j-1] + 1\n                max_len = max(max_len, dp[i][j])\n    return max_len\n```\n\nNow applying this logic to all string pairs from `list1` and `list2`:\n\n### Input:\n```python\nlist1 = [\"hello\", \"world\", \"happiness\"]\nlist2 = [\"howdy\", \"world\", \"happiness\"]\n```\n\nLet's compare each string in `list1` with each string in `list2`, and find the **longest common substring**:\n\n- \"hello\" vs \"howdy\" → common substring: \"h\", length = 1  \n- \"hello\" vs \"world\" → no common substring > 0  \n- \"hello\" vs \"happiness\" → common substring: \"h\", length = 1  \n- \"world\" vs \"howdy\" → common substring: \"ow\", length = 2  \n- \"world\" vs \"world\" → full match, length = 5  \n- \"world\" vs \"happiness\" → no common substring > 0  \n- \"happiness\" vs \"howdy\" → common substring: \"h\", length = 1  \n- \"happiness\" vs \"world\" → no common substring > 0  \n- \"happiness\" vs \"happiness\" → full match, length = 9  \n\n### Final Result:\nThe longest common substring is of length **9** (\"happiness\").\n\n✅ **Answer: `9`**"}]}
    test1 = encode_prompts("", example, tokenizer, 2048)
