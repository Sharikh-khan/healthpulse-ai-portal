# 🩺 HealthPulse AI | Clinical Dashboard & Smart Health Portal

An AI-driven clinical insights, personal health tracking, and patient guidance portal built with **Streamlit** and powered by **Google Gemini**. HealthPulse AI combines hybrid Retrieval-Augmented Generation (RAG), automated lab report OCR/extraction, symptom triage, medication adherence tracking, and live local health alerts in a multi-language interface.

---

## ✨ Key Features

- **📊 Vital Analytics & Smart Alerts**: Interactive Plotly trend charts for Heart Rate, Blood Pressure, and Blood Glucose with automatic clinical threshold alerts[cite: 7].
- **🧪 AI Lab Report Analyzer & OCR**: Upload medical reports (PDFs, text, or scanned images)[cite: 7]. Uses `pypdf` and Gemini Vision OCR to extract numeric test values, flag out-of-range results against reference intervals, and track multi-report trends[cite: 7].
- **💬 RAG-Powered Clinical AI Assistant**: Context-aware health chat grounded in your personal medical records and a curated medical knowledge base using hybrid semantic (`gemini-embedding-001`) and keyword retrieval, complete with source citations and voice input[cite: 7].
- **🩺 Symptom Triage**: Structured urgency assessment (*Self-care*, *See a doctor soon*, or *Emergency*) with red-flag detection, voice-to-text symptom logging, and instant emergency dialing links (`112` / `108`)[cite: 7].
- **💊 Medication & Adherence Tracker**: Daily dose scheduling, allergy conflict warnings, adherence logging, and 7-day adherence analytics[cite: 7].
- **🚨 Emergency Medical ID & QR Code**: Generates a scannable plain-text QR code containing critical patient info (blood group, allergies, ongoing conditions, active medications, and emergency contact)[cite: 7].
- **🗺️ Nearby Care & Live Environmental Insights**: Real-time US AQI and PM2.5/PM10 monitoring via Open-Meteo, seasonal health advisories, and an interactive OpenStreetMap directory of nearby hospitals, clinics, and pharmacies[cite: 7].
- **📅 Appointments & Doctor-Ready Exports**: Schedule appointments with one-click `.ics` calendar sync (`icalendar`) and export comprehensive clinical summary PDFs (`reportlab`)[cite: 7].
- **🌐 Multi-Language Support**: Full AI response translation and localized UI support for **English, Kannada, Hindi, Tamil, and Telugu**[cite: 7].

---

## 🛠️ Tech Stack

- **Frontend & UI**: Streamlit (Custom Dark Glass CSS)[cite: 7]
- **AI & Embeddings**: Google Gemini API (`google-genai` SDK — `gemini-3.8-flash` & `gemini-embedding-001`)[cite: 7]
- **Database & Auth**: SQLite3 with salted `scrypt` password hashing[cite: 7]
- **Data & Visualization**: Pandas, NumPy, Plotly[cite: 7]
- **Document & Utility Processing**: `pypdf`, `reportlab` (PDF generation), `icalendar` (`.ics` export), `qrcode` & `Pillow`[cite: 7]
- **Live External APIs**: Open-Meteo (Geocoding & Air Quality), OpenStreetMap Overpass API[cite: 7]

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have **Python 3.9 or higher** installed on your machine, along with a valid **Google Gemini API Key**[cite: 7].

### 2. Installation
Clone the repository and navigate into the project directory:
```bash
git clone [https://github.com/Sharikh-khan/healthpulse-ai-portal.git](https://github.com/Sharikh-khan/healthpulse-ai-portal.git)
cd healthpulse-ai-portal
