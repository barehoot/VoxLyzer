# VoxLyzer
VoxLyzer Seamlessly analyze sentiment, transcribe audio, scrape URLs, and retrieve information from text with this all-in-one text analysis tool.


# Text Sentiment Analysis Data Tool

![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-red?style=for-the-badge&logo=Streamlit)
![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)

## Introduction

The **Text Sentiment Analysis Data Tool** is a Streamlit-powered application that provides various text analysis functionalities. It allows you to perform sentiment analysis on text, create word clouds, transcribe audio files, and analyze content from different file formats (DOCX, TXT, PDF). This tool is designed to help users gain insights from textual data, making it useful for a wide range of applications, from sentiment analysis to content scraping and analysis.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Features](#features)
4. [Instructions](#instructions)
5. [Contributing](#contributing)
6. [License](#license)

## Installation

To run this Streamlit app, follow these steps:

1. Clone this repository.
2. Navigate to the project's root directory.

   ```bash
   cd Text-Sentiment-Analysis-App
   ```

3. Install the required Python libraries using pip:

   ```bash
   pip install -r requirements.txt
   ```
4. get the ffmpeg for audio processing:
5. 
   '''bash
   sudo apt-get update
   sudo apt-get install ffmpeg
   '''
5. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

The app should open in your default web browser.

## Usage

### Using the App

1. **Entering a News Article URL:**

   - In the sidebar, you can enter a URL of a news article. The app will then fetch the article content and perform analysis.

2. **Uploading Files:**

   - You can upload DOCX, TXT, or PDF files by clicking the "Upload file" button in the sidebar. The app will analyze the content of the uploaded file.

3. **Transcribing Audio:**

   - Upload an audio file (MP3, WAV, or FLAC) by using the "Upload an audio file" button. The app will transcribe the audio and display the transcribed text.

4. **Clearing Inputs:**

   - Click the "Clear All" button to reset all inputs and start over.

For detailed instructions on using the app and understanding the results, refer to [instructions.txt](instructions.txt).

## Features

- **Sentiment Analysis:** The app provides sentiment analysis with charts indicating positive, negative, neutral, and compound scores.

- **Word Cloud:** Generate word clouds to visualize the most frequent words in the provided text.

- **File Format Support:** Analyze content from DOCX, TXT, and PDF files, making it versatile for different document types.

- **Audio Transcription:** Transcribe audio files to text, expanding the app's capabilities.

- **Content Scraping:** Extract and analyze content from news article URLs.

## Instructions

For detailed instructions on using the app and understanding the results, please refer to [instructions.txt](instructions.txt) included in the project's root directory.

## Contributing

If you'd like to contribute to this project, please fork the repository and create a pull request. We welcome contributions that enhance the functionality and usability of the app.

## License

This project is licensed under the [MIT License](LICENSE).

---

Feel free to use this template for your README file. You can add or modify sections as needed to provide more information about your specific app and project.
