import google.generativeai as genai
import ast
import re
import json
import time

genai.configure(api_key='')

path = "training_data/pro"
# suffix_list = ['', '_CoT', '_ToT']
suffix_list = ['', '_CoT', '_ToT', '_IRSfix', '_IRSori', '_POP', '_random']
_suf = suffix_list[5]
_model_suf = '_Gemini'

with open(f"{path}/output_gpt_1m{_suf}_pro.json", "r", encoding='utf-8') as f:
    path_result = json.load(f)
with open("training_data/train_1m.json", "r", encoding='utf-8') as f:
    training_data_prompt = json.load(f)

prompt_system_accept = "You are a recommender system. Given the user profile and historical data, analyze the user's interests. Based on this information, would the user be interested in watching the movies in the Influence path step by step? Answer with a probability for a movie between 0 and 1, where 0 means 'definitely not interested' and 1 means 'definitely interested'. If uncertain, make your best guess, then return the mean probability."

prompt_system_relevance = "You are a professional movie critic. Given the Influence path, based on your understanding of the movies, what's the relatedness of each 2 adjacent movies in the influence path? Answer with a probability between 0 and 1, where 1 means 'definitely related' and 0 means 'definitely not related'. If uncertain, make your best guess, then return the mean probability."


def get_score(_question, _prompt):
    if _prompt == []:
        return [0, 0]
    else:
        influence_path = str.join("\n", _prompt)

    _question = _question[:_question.index("Target movie:")]
    user_prompt = f"{_question}Influence path:\n{influence_path}"

    # print(user_prompt)

    loop_flag = 0
    while True:
        time.sleep(10)
        message = [
            {"role": "user", "parts": [prompt_system_relevance, user_prompt]}
        ]
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(message)
        # print(prompt_system_accept)
        # print(user_prompt)
        # print(response.text)

        try:
            gpt_reply = response.text
        except:
            print(f"message: {user_prompt}")
            print(f"Error: {response}")
            continue

        time.sleep(10)
        message.append({"role": "model", "parts": [gpt_reply]})
        message.append({"role": "user",
                        "parts": [
                            "Only numerical response required, without any other word. Output the mean probability."]})

        response = model.generate_content(message)

        try:
            gpt_reply = response.text
        except:
            print(f"message: {user_prompt}")
            print(f"Error: {response}")
            continue
        # print(gpt_reply)

        if re.search('0\.\d+', gpt_reply):
            return [float(re.search('0\.\d+', gpt_reply).group()), gpt_reply]

        loop_flag += 1
        if loop_flag >= 2:
            print(gpt_reply)


k = 1
all_result = []
for prefix, influ_path in zip(training_data_prompt['llm'], path_result):
    print(k)
    k += 1
    all_result.append([get_score(prefix[1], influ_path[0]), get_score(prefix[1], influ_path[1])])
    # print(all_result)
    if k == 20:
        break
    # break

with open(f"{path}/llm_result_1m{_suf}_rel{_model_suf}.json", "w") as outfile:
    json.dump(all_result, outfile)

sum = 0
count = 0
for _ in all_result:

    a = _[0][0]
    b = _[1][0]
    max_ = max(a, b)
    if max_ == 0:
        continue
    sum += max_
    count += 1

print(_suf)
print(sum / count)
