# Say Less - Make Decisions with Confidence

Say Less is a minimalist web application designed to help you make better decisions by simplifying your decision-making process. Whether you're choosing between job offers, deciding on a purchase, or figuring out your next life move, Say Less provides the structure and clarity you need.

## Features

- 🔍 **Widen Your Options**: Explore alternatives you might not have considered
- 🎯 **Sharpen Your Criteria**: Identify what truly matters in your decision
- ⚡ **Decide Faster**: Cut through analysis paralysis with AI-powered insights
- 🌓 **Dark Mode Support**: Easy on the eyes, day or night
- 📱 **Mobile-Friendly**: Make decisions on the go

## Getting Started

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Mobile App Development

```bash
# Install Capacitor
npm install @capacitor/core @capacitor/cli
npm install @capacitor/android @capacitor/ios

# Initialize Capacitor
npx cap init

# Build the web app
npm run build

# Add platforms
npx cap add android
npx cap add ios

# Sync web code to native projects
npx cap sync

# Generate app icons and splash screens
npx capacitor-assets generate
```

## Technologies Used

- **Alpine.js**: For reactive UI components
- **Tailwind CSS**: For utility-first styling
- **Marked.js**: For markdown rendering
- **HTMX**: For enhanced interactions
- **Capacitor**: For native mobile capabilities

npm init -y
npm install @capacitor/core @capacitor/cli
npm install @capacitor/android @capacitor/ios
npm install @revenuecat/purchases-capacitor
npm install serve --save-dev
npx cap init
npm run build
npx cap add android
npx cap add ios
npm run build
npm run start
npx cap sync
npx capacitor-assets generate