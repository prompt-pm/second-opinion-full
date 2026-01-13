import globals from "globals";

export default [
  {
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "module",
      globals: {
        ...globals.browser,
        Alpine: "readonly",
      },
    },
    rules: {
      // Possible errors
      "no-console": "warn",
      "no-debugger": "error",
      "no-duplicate-case": "error",
      "no-empty": "error",
      "no-extra-boolean-cast": "error",
      "no-func-assign": "error",
      "no-irregular-whitespace": "error",
      "no-unreachable": "error",

      // Best practices
      "curly": ["error", "multi-line"],
      "eqeqeq": ["error", "always", { "null": "ignore" }],
      "no-eval": "error",
      "no-implied-eval": "error",
      "no-multi-spaces": "error",
      "no-return-assign": "error",
      "no-unused-expressions": "error",
      "no-useless-concat": "error",

      // Variables
      "no-undef": "error",
      "no-unused-vars": ["error", { "argsIgnorePattern": "^_" }],
      "no-use-before-define": ["error", { "functions": false }],

      // Stylistic
      "indent": ["error", 4],
      "quotes": ["error", "single", { "avoidEscape": true }],
      "semi": ["error", "always"],
      "comma-dangle": ["error", "always-multiline"],
      "no-trailing-spaces": "error",
    },
  },
  {
    files: ["**/*.test.js", "**/*.spec.js", "tests/**/*.js"],
    languageOptions: {
      globals: {
        ...globals.node,
        describe: "readonly",
        it: "readonly",
        expect: "readonly",
        beforeEach: "readonly",
        afterEach: "readonly",
        vi: "readonly",
        global: "readonly",
      },
    },
    rules: {
      "no-console": "off",
    },
  },
  {
    ignores: ["node_modules/**", ".context/**", ".venv/**", "backend/**", "config/**"],
  },
];
