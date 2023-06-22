# Palabrar: A Spanish Learning Platform

Palabrar is an innovative Spanish learning platform that combines the essence of Tofugu and WaniKani for Spanish language learning. Our platform uses Natural Language Processing (NLP) to enhance user experience by providing tailored vocabulary lists and resource recommendations based on user input. 

## Features

- Customized Vocabulary Builder with Spaced Repetition
- In-depth Grammar Lessons and Exercises
- Phonetics Lessons including Consonants, Vowels, Diphthongs, and more
- Resources Section with Links to Podcasts, Articles, and Interviews
- Interactive Tools such as Flashcards and Conjugators
- User Authentication and Profile Management

## Tech Stack

- Frontend: Next.js
- Backend: Serverless Functions on Vercel
- Machine Learning: Python, PyTorch/TensorFlow
- Database: Firestore (Google Firebase)
- Authentication: Firebase Authentication
- Content Management: Contentful/Strapi
- Hosting: Vercel and Firebase

## Getting Started

### Prerequisites

- Node.js
- npm
- Python (for NLP model)
- Firebase Account

### Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/palabrar.git
    cd palabrar
    ```

2. Install the dependencies:

    ```sh
    npm install
    ```

3. Set up Firebase (Follow Firebase documentation to setup Firestore and Authentication).

4. Set environment variables for Firebase configuration.

5. Start the development server:

    ```sh
    npm run dev
    ```

6. Open `http://localhost:3000` in your browser.

## Deployment

The project is ready to be deployed on Vercel. Follow [Vercel's Deployment Guide](https://vercel.com/docs/deployments).

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT

## Acknowledgements

- Professor Wei Xu for his invaluable guidance.
- Octavio Calvo for his assistance with Spanish interpretation and structure.

