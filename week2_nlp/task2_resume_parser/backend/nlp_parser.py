# ============================================================
# WEEK 2 - TASK 02
# RESUME PARSER USING NLP
# ROBUST VERSION
# ============================================================

import re
import json
import spacy


# ============================================================
# LOAD SPACY
# ============================================================

nlp = spacy.load("en_core_web_sm")


# ============================================================
# SKILLS
# ============================================================

SKILLS = [
    "Python", "Java", "C", "C++", "C#", "JavaScript",
    "TypeScript", "PHP", "R",
    "HTML", "CSS", "React", "Angular", "Node.js",
    "Express", "Django", "Flask",
    "MySQL", "MongoDB", "PostgreSQL", "SQL", "Oracle",
    "Machine Learning", "Deep Learning",
    "Artificial Intelligence", "NLP",
    "Natural Language Processing", "TensorFlow",
    "PyTorch", "Scikit-learn", "spaCy", "NLTK",
    "Data Science", "Data Analysis", "Pandas",
    "NumPy", "Matplotlib",
    "AWS", "Azure", "Google Cloud",
    "Git", "GitHub", "Docker",
    "Cybersecurity", "Network Security",
    "Ethical Hacking", "Blockchain",
    "Communication", "Leadership", "Networking",
    "Project Management", "Strategic Planning",
    "Team Leadership"
]


# ============================================================
# SECTION KEYWORDS
# ============================================================

SECTION_KEYWORDS = {
    "summary": [
        "summary",
        "professional summary",
        "profile"
    ],

    "skills": [
        "skills",
        "technical skills",
        "core skills",
        "areas of expertise",
        "key skills"
    ],

    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment",
        "career history"
    ],

    "education": [
        "education",
        "academic background",
        "academic qualifications",
        "educational background"
    ],

    "certifications": [
        "certifications",
        "certificates",
        "courses",
        "training"
    ],

    "projects": [
        "projects",
        "academic projects",
        "personal projects"
    ],

    "languages": [
        "languages",
        "language"
    ]
}


# ============================================================
# NORMALIZE LINE
# ============================================================

def normalize_line(line):

    line = line.strip()

    line = re.sub(
        r"^[•●▪■\-]+\s*",
        "",
        line
    )

    line = re.sub(
        r"^\d+[\.\)]\s*",
        "",
        line
    )

    line = line.strip(
        " :|"
    )

    return line


# ============================================================
# DETECT SECTION
# ============================================================

def detect_section(line):

    cleaned = normalize_line(line).lower()

    # Direct match
    for section, names in SECTION_KEYWORDS.items():

        for name in names:

            if cleaned == name:
                return section

    # Heading embedded in line
    for section, names in SECTION_KEYWORDS.items():

        for name in names:

            if (
                cleaned.startswith(name + " ")
                or cleaned.startswith(name + ":")
            ):
                return section

    return None


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    text = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f]",
        " ",
        text
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# EXTRACT SECTIONS
# ============================================================

def extract_sections(text):

    sections = {
        "general": []
    }

    current = "general"

    lines = text.split("\n")

    for raw_line in lines:

        line = normalize_line(raw_line)

        if not line:
            continue

        section = detect_section(line)

        if section:

            current = section

            if current not in sections:
                sections[current] = []

            continue

        sections.setdefault(
            current,
            []
        ).append(line)

    # Convert lists to strings
    for key in sections:

        sections[key] = "\n".join(
            sections[key]
        )

    return sections


# ============================================================
# EMAIL
# ============================================================

def extract_email(text):

    match = re.search(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        text
    )

    return match.group(0) if match else ""


# ============================================================
# PHONE
# ============================================================

def extract_phone(text):

    patterns = [

        r"\(\d{3}\)\s*\d{3}[-.\s]\d{4}",

        r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b",

        r"\+\d{1,3}[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}"

    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text
        )

        if match:
            return match.group(0)

    return ""


# ============================================================
# LINKEDIN
# ============================================================

def extract_linkedin(text):

    match = re.search(
        r"(?:https?://)?(?:www\.)?linkedin\.com/[^\s]+",
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(0).rstrip(
            ".,)"
        )

    return ""


# ============================================================
# GITHUB
# ============================================================

def extract_github(text):

    match = re.search(
        r"(?:https?://)?(?:www\.)?github\.com/[^\s]+",
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(0).rstrip(
            ".,)"
        )

    return ""


# ============================================================
# NAME
# ============================================================

def extract_name(text):

    lines = [
        normalize_line(line)
        for line in text.split("\n")
        if normalize_line(line)
    ]

    for line in lines[:10]:

        if "@" in line:
            continue

        if re.search(
            r"\d",
            line
        ):
            continue

        words = line.split()

        if not 2 <= len(words) <= 4:
            continue

        if detect_section(line):
            continue

        if re.fullmatch(
            r"[A-Za-z][A-Za-z .'-]+",
            line
        ):

            return line

    doc = nlp(
        text[:1500]
    )

    for ent in doc.ents:

        if ent.label_ == "PERSON":

            name = ent.text.strip()

            if 2 <= len(name.split()) <= 4:
                return name

    return ""


# ============================================================
# SKILLS
# ============================================================

def extract_skills(text):

    found = []

    lower_text = text.lower()

    for skill in SKILLS:

        pattern = (
            r"(?<![A-Za-z0-9])"
            + re.escape(skill.lower())
            + r"(?![A-Za-z0-9])"
        )

        if re.search(
            pattern,
            lower_text
        ):

            found.append(skill)

    return sorted(
        list(set(found)),
        key=str.lower
    )


# ============================================================
# EXPERIENCE
# ============================================================

def extract_experience(sections):

    experience = []

    text = sections.get(
        "experience",
        ""
    )

    if not text:
        return experience

    job_keywords = [
        "manager",
        "developer",
        "engineer",
        "intern",
        "analyst",
        "designer",
        "consultant",
        "administrator",
        "specialist",
        "executive",
        "assistant",
        "director",
        "lead",
        "officer",
        "scientist",
        "architect",
        "coordinator",
        "associate"
    ]

    # Examples:
    # Sep 2022 - Apr 2025
    # Dec 2018 - Aug 2022
    # 2018 - 2022

    date_pattern = re.compile(
        r"\b(?:"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"\.?\s+\d{4}"
        r"\s*[-–]\s*"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"\.?\s+\d{4}"
        r"|"
        r"\d{4}\s*[-–]\s*\d{4}"
        r")\b",
        re.IGNORECASE
    )

    for line in text.split("\n"):

        line = normalize_line(line)

        if not line:
            continue

        lower = line.lower()

        has_job_title = any(
            keyword in lower
            for keyword in job_keywords
        )

        has_date = bool(
            date_pattern.search(line)
        )

        # Strongest signal:
        # job title + date
        if has_job_title and has_date:

            if line not in experience:
                experience.append(line)

    return experience


# ============================================================
# EDUCATION
# ============================================================

def extract_education(sections):

    education = []

    text = sections.get(
        "education",
        ""
    )

    keywords = [
        "bachelor",
        "master",
        "degree",
        "b.e",
        "b.tech",
        "m.e",
        "m.tech",
        "b.sc",
        "m.sc",
        "bca",
        "mca",
        "mba",
        "phd",
        "diploma",
        "university",
        "college"
    ]

    for line in text.split("\n"):

        line = normalize_line(line)

        if not line:
            continue

        if any(
            keyword in line.lower()
            for keyword in keywords
        ):

            if line not in education:
                education.append(line)

    return education


# ============================================================
# CERTIFICATIONS
# ============================================================

def extract_certifications(sections):

    certifications = []

    text = sections.get(
        "certifications",
        ""
    )

    for line in text.split("\n"):

        line = normalize_line(line)

        if not line:
            continue

        if line not in certifications:

            certifications.append(line)

    return certifications


# ============================================================
# PROJECTS
# ============================================================

def extract_projects(sections):

    projects = []

    text = sections.get(
        "projects",
        ""
    )

    for line in text.split("\n"):

        line = normalize_line(line)

        if line and line not in projects:

            projects.append(line)

    return projects


# ============================================================
# LANGUAGES
# ============================================================

def extract_languages(sections):

    languages = []

    text = sections.get(
        "languages",
        ""
    )

    common_languages = [
        "English",
        "Spanish",
        "French",
        "German",
        "Hindi",
        "Kannada",
        "Telugu",
        "Tamil",
        "Marathi",
        "Malayalam",
        "Urdu",
        "Arabic",
        "Chinese",
        "Japanese"
    ]

    for line in text.split("\n"):

        line = normalize_line(line)

        if not line:
            continue

        for language in common_languages:

            if line.lower() == language.lower():

                if language not in languages:
                    languages.append(language)

    return languages


# ============================================================
# NER
# ============================================================

def extract_entities(text):

    doc = nlp(text)

    entities = []

    allowed_labels = {
        "PERSON",
        "ORG",
        "GPE",
        "FAC",
        "LANGUAGE"
    }

    for ent in doc.ents:

        value = ent.text.strip()

        if ent.label_ not in allowed_labels:
            continue

        if len(value) < 2:
            continue

        # Remove bullet contamination
        value = re.sub(
            r"^[•●▪■\-]+\s*",
            "",
            value
        ).strip()

        if not value:
            continue

        # Ignore known noisy patterns
        if re.search(
            r"\b(?:LinkedIn|Manage|Secure|Successfully|Organized)\b",
            value,
            re.IGNORECASE
        ):
            continue

        # Ignore address fragments
        if re.fullmatch(
            r"(?:TX|CA)\s*\d*",
            value,
            re.IGNORECASE
        ):
            continue

        entities.append({
            "text": value,
            "label": ent.label_
        })

    # Remove duplicates
    unique = []
    seen = set()

    for entity in entities:

        key = (
            entity["text"].lower(),
            entity["label"]
        )

        if key not in seen:

            seen.add(key)
            unique.append(entity)

    return unique


# ============================================================
# MAIN PARSER
# ============================================================

def parse_resume(text):

    text = clean_text(text)

    sections = extract_sections(text)

    return {

        "personal_information": {

            "name": extract_name(text),

            "email": extract_email(text),

            "phone": extract_phone(text),

            "linkedin": extract_linkedin(text),

            "github": extract_github(text)
        },

        "skills": extract_skills(text),

        "education": extract_education(
            sections
        ),

        "experience": extract_experience(
            sections
        ),

        "certifications": extract_certifications(
            sections
        ),

        "projects": extract_projects(
            sections
        ),

        "languages": extract_languages(
            sections
        ),

        "named_entities": extract_entities(
            text
        )
    }


# ============================================================
# DISPLAY
# ============================================================

def display_result(result):

    print("\n" + "=" * 60)
    print("RESUME NLP PARSER TEST")
    print("=" * 60)

    print("\nPERSONAL INFORMATION")
    print("-" * 60)

    for key, value in result[
        "personal_information"
    ].items():

        print(
            f"{key}: {value}"
        )

    print("\nSKILLS")
    print("-" * 60)

    for item in result["skills"]:
        print(f"• {item}")

    print("\nEDUCATION")
    print("-" * 60)

    for item in result["education"]:
        print(f"• {item}")

    print("\nEXPERIENCE")
    print("-" * 60)

    for item in result["experience"]:
        print(f"• {item}")

    print("\nCERTIFICATIONS")
    print("-" * 60)

    for item in result["certifications"]:
        print(f"• {item}")

    print("\nPROJECTS")
    print("-" * 60)

    for item in result["projects"]:
        print(f"• {item}")

    print("\nLANGUAGES")
    print("-" * 60)

    for item in result["languages"]:
        print(f"• {item}")

    print("\nNAMED ENTITIES")
    print("-" * 60)

    for entity in result["named_entities"]:

        print(
            f"• {entity['text']} → "
            f"{entity['label']}"
        )

    print("\n" + "=" * 60)
    print("NLP PARSING COMPLETED")
    print("=" * 60)


# ============================================================
# SAVE JSON
# ============================================================

def save_json(
    result,
    filename="parsed_resume.json"
):

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"\nJSON saved successfully: {filename}"
    )


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("RESUME NLP PARSER TEST")
    print("=" * 60)

    resume_path = input(
        "\nEnter resume file path: "
    ).strip()

    try:

        from text_extractor import extract_text

        resume_text = extract_text(
            resume_path
        )

        result = parse_resume(
            resume_text
        )

        display_result(result)

        save_json(result)

    except Exception as error:

        print("\nERROR:")
        print(error)