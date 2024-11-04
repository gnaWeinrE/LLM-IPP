import google.generativeai as genai
import ast
import re
import json
import time

genai.configure(api_key='')

path = "training_data/lastfm"
# suffix_list = ['', '_CoT', '_ToT']
suffix_list = ['', '_CoT', '_ToT', '_IRSfix', '_IRSori', '_POP']
# _suf = suffix_list[5]
_model_suf = '_Gemini'

prompt_system_accept = "You are a recommender system. Given the user profile and historical data, analyze the user's interests. Based on this information, would the user be interested in the music artists in the Influence path step by step? Answer with a probability for an artist between 0 and 1, where 0 means 'definitely not interested' and 1 means 'definitely interested'. If uncertain, make your best guess, then return the mean probability."

prompt_system_relevance = "You are a professional music critic. Given the Influence path, based on your understanding of the music artists, what's the relatedness of each 2 adjacent artists in the influence path? Answer with a probability between 0 and 1, where 1 means 'definitely related' and 0 means 'definitely not related'. If uncertain, make your best guess, then return the mean probability."

# system_prompt_list = [[prompt_system_accept, '_accept'], [prompt_system_relevance, '_rel']]
system_prompt_list = [[prompt_system_relevance, '_rel']]

def get_score(_question, _prompt, _system_prompt):
    if _prompt == []:
        return [0, 0]
    else:
        influence_path = str.join("\n", _prompt)

    user_prompt = f"{_question}Influence path:\n{influence_path}"

    # print(user_prompt)

    loop_flag = 0
    while True:
        time.sleep(10)
        message = [
            {"role": "user", "parts": [_system_prompt, user_prompt]}
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
        elif re.search('\d', gpt_reply):
            return [float(re.search('\d', gpt_reply).group()), gpt_reply]

        loop_flag += 1
        if loop_flag >= 2:
            print(gpt_reply)


for system_prompt, system_prompt_suffix in system_prompt_list:
    print(system_prompt_suffix)
    for _suf in suffix_list:
        print(_suf)

        with open(f"{path}/output_gpt_lf{_suf}_pro.json", "r", encoding='utf-8') as f:
            path_result = json.load(f)
        with open("training_data/train_lf.json", "r", encoding='utf-8') as f:
            training_data_prompt = json.load(f)

        k = 1
        all_result = []
        for prefix, influ_path in zip(training_data_prompt['llm'], path_result):
            if k%20 == 0:
                print(k)
            k += 1
            all_result.append([get_score(prefix[1], influ_path[0], system_prompt), get_score(prefix[1], influ_path[1], system_prompt)])
            # print(all_result)
            if k == 20:
                break
            # break

        with open(f"{path}/llm_result_lf{_suf}{system_prompt_suffix}{_model_suf}.json", "w") as outfile:
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
