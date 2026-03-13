# Dual Mode Mentor Bot / 5Mini Quiz & Progress App

This project is a web-based learning platform developed as a group project.  
It helps students practice quizzes, track their learning progress, and interact with a mentor bot for guidance.

## Technologies Used

Frontend:
- HTML
- CSS
- JavaScript

Backend:
- Python
- Flask

Database / Storage:
- Firebase

AI Integration:
- Gemini API (for quiz generation)

## Features

- Interactive quiz system
- Student progress tracking
- Study room quiz feature
- Mentor chatbot support
- AI-assisted quiz generation using Gemini
- Firebase-based storage for user data
- Web-based learning environment

## How to Run the Project

### Prerequisites
- Python 3.10 or higher

Install dependencies:

pip install -r requirements.txt

Run the Flask backend:

python app.py

Open the application in a browser:

http://127.0.0.1:5000/index.html

## Notes

- If `GEMINI_API_KEY` is added in the `.env` file, the system generates quizzes using Gemini AI.
- If the API key is not available or the internet is unavailable, the system automatically uses a local quiz generator.
- User progress is stored using Firebase and session-based synchronization.

## Project Structure

- Frontend: HTML, CSS, JavaScript
- Backend: Python (Flask)
- Storage: Firebase

## Developed By

Pranjali Shetty and Team
