from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RESULT_DIR = os.getenv("RESULT_DIR", "results")

LANGUAGES = {
    "Eastern Iranian - Northeastern": {
        "Ossetic": [
            "(Russian Federation)"
        ],
        "Yagnobi": [
            "(Tajikistan)"
        ],
        "Yassic": [
            "(Hungary)"
        ]
    },
    "Eastern Iranian - Southeastern": {
        "Pamir": [
            "Ishkashimi (Afghanistan)",
            "Munji (Afghanistan)",
            "Sanglechi (Afghanistan)",
            "Wakhi (Afghanistan)",
            "Yadgha (Pakistan)",
            "Shugni-Yazgulami Sarikoli (China)",
            "Shugni-Yazgulami Shughni (Tajikistan)",
            "Shugni-Yazgulami Yazghulami (Tajikistan)"
        ],
        "Pashto": [
            "Pashto, Central (Pakistan)",
            "Pashto, Northern (Pakistan)",
            "Pashto, Southern (Afghanistan)",
            "Waneci (Pakistan)"
        ]
    },
    "Western Iranian - Northwestern": {
        "Balochi": [
            "Balochi, Eastern (Pakistan)",
            "Balochi, Southern (Pakistan)",
            "Balochi, Western (Pakistan)",
            "Bashkardi (Iran)",
            "Koroshi (Iran)"
        ],
        "Caspian": [
            "Gilaki (Iran)",
            "Mazandarani (Iran)",
            "Shahmirzadi (Iran)"
        ],
        "Central Iran": [
            "Ashtiani (Iran)",
            "Dari, Zoroastrian (Iran)",
            "Gazi (Iran)",
            "Khunsari (Iran)",
            "Natanzi (Iran)",
            "Nayini (Iran)",
            "Parsi-Dari (Iran)",
            "Sivandi (Iran)",
            "Soi (Iran)",
            "Vafsi (Iran)"
        ],
        "Kurdish": [
            "Kurdish, Central (Iraq)",
            "Kurdish, Northern (Turkey)",
            "Kurdish, Southern (Iran)",
            "Laki (Iran)"
        ],
        "Ormuri-Parachi": [
            "Ormuri (Pakistan)",
            "Parachi (Afghanistan)"
        ],
        "Semnani": [
            "Lasgerdi (Iran)",
            "Sangisari (Iran)",
            "Semnani (Iran)",
            "Sorkhei (Iran)"
        ],
        "Talysh": [
            "Alviri-Vidari (Iran)",
            "Eshtehardi (Iran)",
            "Gozarkhani (Iran)",
            "Harzani (Iran)",
            "Kabatei (Iran)",
            "Kajali (Iran)",
            "Karingani (Iran)",
            "Kho'ini (Iran)",
            "Koresh-e Rostam (Iran)",
            "Maraghei (Iran)",
            "Razajerdi (Iran)",
            "Rudbari (Iran)",
            "Shahrudi (Iran)",
            "Takestani (Iran)",
            "Talysh (Azerbaijan)",
            "Taromi, Upper (Iran)"
        ],
        "Zaza-Gorani": [
            "Bajelani (Iraq)",
            "Gurani (Iran)",
            "Kakayi (Iraq)",
            "Shabak (Iraq)",
            "Zazaki, Northern (Turkey)",
            "Zazaki, Southern (Turkey)"
        ],
        "Unclassified": [
            "Dezfuli (Iran)"
        ]
    },
    "Western Iranian - Southwestern": {
        "Fars": [
            "Fars, Southwestern (Iran)",
            "Lari (Iran)"
        ],
        "Luri": [
            "Bakhtiari (Iran)",
            "Laki (Iran)",
            "Luri, Northern (Iran)",
            "Luri, Southern (Iran)"
        ],
        "Persian": [
            "Aimaq (Afghanistan)",
            "Bukharic (Uzbekistan)",
            "Dari (Afghanistan)",
            "Dehwari (Pakistan)",
            "Hazaragi (Afghanistan)",
            "Judeo-Persian (Iran)",
            "Pahlavani (Afghanistan)",
            "Persian, Iranian (Iran)",
            "Tajik (Tajikistan)"
        ],
        "Tat": [
            "Judeo-Tat (Russian Federation)",
            "Tat, Muslim (Azerbaijan)"
        ]
    }
}