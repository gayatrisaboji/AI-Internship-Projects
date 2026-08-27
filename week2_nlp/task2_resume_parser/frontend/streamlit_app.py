import streamlit as st
import requests
import json


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Resume NLP Parser",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📄 Resume NLP Parser")
st.write(
    "Upload a resume PDF and automatically extract "
    "personal information, skills, education, experience, "
    "certifications, projects, languages and named entities."
)

st.divider()


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)


# ============================================================
# PARSE RESUME
# ============================================================

if uploaded_file is not None:

    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("🔍 Parse Resume", type="primary"):

        try:
            files = {
                "resume": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/pdf"
                )
            }

            with st.spinner("Parsing resume..."):

                response = requests.post(
                    "http://127.0.0.1:5000/parse-resume",
                    files=files
                )

            if response.status_code == 200:

                result = response.json()

                if result.get("status") == "success":

                    data = result.get("data", {})

                    st.success("✅ Resume parsed successfully!")

                    # ====================================================
                    # PERSONAL INFORMATION
                    # ====================================================

                    st.header("👤 Personal Information")

                    personal = data.get(
                        "personal_information",
                        {}
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(
                            f"**Name:** "
                            f"{personal.get('name', '')}"
                        )

                        st.write(
                            f"**Email:** "
                            f"{personal.get('email', '')}"
                        )

                        st.write(
                            f"**Phone:** "
                            f"{personal.get('phone', '')}"
                        )

                    with col2:
                        st.write(
                            f"**LinkedIn:** "
                            f"{personal.get('linkedin', '') or 'Not found'}"
                        )

                        st.write(
                            f"**GitHub:** "
                            f"{personal.get('github', '') or 'Not found'}"
                        )

                    # ====================================================
                    # SKILLS
                    # ====================================================

                    st.header("🛠️ Skills")

                    skills = data.get("skills", [])

                    if skills:
                        for skill in skills:
                            st.markdown(f"• {skill}")
                    else:
                        st.info("No skills detected.")

                    # ====================================================
                    # EDUCATION
                    # ====================================================

                    st.header("🎓 Education")

                    education = data.get("education", [])

                    if education:
                        for item in education:
                            st.markdown(f"• {item}")
                    else:
                        st.info("No education information detected.")

                    # ====================================================
                    # EXPERIENCE
                    # ====================================================

                    st.header("💼 Experience")

                    experience = data.get("experience", [])

                    if experience:
                        for item in experience:
                            st.markdown(f"• {item}")
                    else:
                        st.info("No experience information detected.")

                    # ====================================================
                    # CERTIFICATIONS
                    # ====================================================

                    st.header("🏆 Certifications")

                    certifications = data.get(
                        "certifications",
                        []
                    )

                    if certifications:
                        for item in certifications:
                            st.markdown(f"• {item}")
                    else:
                        st.info("No certifications detected.")

                    # ====================================================
                    # PROJECTS
                    # ====================================================

                    st.header("🚀 Projects")

                    projects = data.get("projects", [])

                    if projects:
                        for item in projects:
                            st.markdown(f"• {item}")
                    else:
                        st.info("No projects detected.")

                    # ====================================================
                    # LANGUAGES
                    # ====================================================

                    st.header("🌐 Languages")

                    languages = data.get("languages", [])

                    if languages:
                        for language in languages:
                            st.markdown(f"• {language}")
                    else:
                        st.info("No languages detected.")

                    # ====================================================
                    # NAMED ENTITIES
                    # ====================================================

                    st.header("🔎 Named Entities")

                    entities = data.get(
                        "named_entities",
                        []
                    )

                    if entities:

                        entity_rows = []

                        for entity in entities:
                            entity_rows.append(
                                {
                                    "Text": entity.get(
                                        "text",
                                        ""
                                    ),
                                    "Label": entity.get(
                                        "label",
                                        ""
                                    )
                                }
                            )

                        st.dataframe(
                            entity_rows,
                            use_container_width=True
                        )

                    else:
                        st.info("No named entities detected.")

                    # ====================================================
                    # JSON OUTPUT
                    # ====================================================

                    st.header("📋 JSON Output")

                    st.json(data)

                    # ====================================================
                    # DOWNLOAD JSON
                    # ====================================================

                    json_data = json.dumps(
                        data,
                        indent=4
                    )

                    st.download_button(
                        label="⬇️ Download Parsed JSON",
                        data=json_data,
                        file_name="parsed_resume.json",
                        mime="application/json"
                    )

                else:

                    st.error(
                        result.get(
                            "message",
                            "Resume parsing failed."
                        )
                    )

            else:

                st.error(
                    f"API Error: HTTP {response.status_code}"
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "❌ Could not connect to Flask API. "
                "Please start the Flask backend first."
            )

        except Exception as e:

            st.error(
                f"❌ Error: {str(e)}"
            )