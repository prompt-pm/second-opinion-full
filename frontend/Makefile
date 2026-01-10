-include .env
export

.PHONY: dev
dev:
	npm run dev

.PHONY: build
build:
	npm install
	npm run build

.PHONY: ios
ios:
	npm run build
	npx cap sync
	npx cap run ios --livereload

.PHONY: ios-simulator
ios-simulator:
	npm run build
	npx cap sync
	npx cap run ios --livereload --target "8160CEAA-96C2-4A24-BD24-BB67B7636DDA"

.PHONY: android
android:
	npm run build
	npx cap sync
	npx cap run android

.PHONY: ios-deploy
ios-deploy:
	npx cap open ios

.PHONY: android-deploy
android-deploy:
	cd android && ./gradlew build
	cd ../
	npx cap open android

.PHONY: bundle
bundle:
	npm run build
	npx cap sync
	npx @capgo/cli bundle zip