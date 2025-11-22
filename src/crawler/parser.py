import json
from openai import OpenAI


def check_json_validity(json_string: str) -> bool:
    try:
        data = json.loads(json_string)
        
        # Check if data is a list
        if isinstance(data, list):
            # Validate each item in the list
            for item in data:
                if not isinstance(item, dict):
                    return False, None, "Data should be a list of objects"
                if 'original' not in item or 'translation' not in item:
                    return False, None, "Each object must have 'original' and 'translation' fields"
            return True, data, ""
        
        # Check if data is a dict with a 'data' key
        if isinstance(data, dict):
            if 'data' in data:
                actual_data = data['data']
                if not isinstance(actual_data, list):
                    return False, None, "'data' field should contain a list"
                for item in actual_data:
                    if not isinstance(item, dict):
                        return False, None, "Data should be a list of objects"
                    if 'original' not in item or 'translation' not in item:
                        return False, None, "Each object must have 'original' and 'translation' fields"
                return True, actual_data, ""
            
            if len(data) == 0:
                return True, [], ""
            # If dict has only one key, try to use its value
            if len(data) == 1:
                actual_data = list(data.values())[0]
                if not isinstance(actual_data, list):
                    return False, None, "Expected a list of objects"
                for item in actual_data:
                    if not isinstance(item, dict):
                        return False, None, "Data should be a list of objects"
                    if 'original' not in item or 'translation' not in item:
                        return False, None, "Each object must have 'original' and 'translation' fields"
                return True, actual_data, ""
        
        return False, None, "Data should be a list or dict containing a list of objects with 'original' and 'translation' fields"
        
    except json.JSONDecodeError as e:
        return False, None, f"Invalid JSON: {str(e)}"


def extract_samples(client: OpenAI, content: str, language: str, dialect: str, accent: str, target_language: str, history_messages: list = None) -> list[str]:
    if history_messages:
        messages = history_messages
    else:
        system_prompt = (
            "You are an expert in extracting language samples from texts."
            "Given a text, you will extract any sentences or phrases that exemplify a specific language, dialect, and accent."
            "]"
        )
        user_prompt = (
            f"Extract any samples of sentences or phrases for {language} language, {dialect} dialect, {accent} accent from the following text:\n\n{content}"
            f"Also if there is translations available in the {target_language}, include them as well in a json array format."
            "format the output as a json array of objects with 'original' and 'translation' fields."
            "data: "
            "["
            "{'original': '<original sentence>', 'translation': '<translation in target language>'}"
            "..."
            "]"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    response = client.chat.completions.create(
        model="gpt-5",
        messages=messages,
        response_format={"type": "json_object"},
    )

    is_valid, data, error_message = check_json_validity(response.choices[0].message.content)
    if is_valid:
        return data
    else:
        print(f"JSON validation error: {error_message}. Retrying extraction...")
        history_messages = messages + [
            {"role": "assistant", "content": response.choices[0].message.content},
            {"role": "user", "content": error_message}
        ]
        return extract_samples(client, content, language, dialect, accent, target_language, history_messages=history_messages)

def automatic_validation(samples: list[dict[str, str]], source_context: str) -> list[dict[str, str]]:
    for sample in samples:
        original_check = sample['original'] in source_context
        if sample['translation']:
            translation_check = sample['translation'] in source_context
        else:
            translation_check = True
        sample['original_check'] = original_check
        sample['translation_check'] = translation_check
    return samples
                        