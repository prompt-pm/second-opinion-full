/**
 * Second Opinion - Frontend Application Logic
 * Extracted for testing purposes
 */

export function createAppData() {
    return {
        view: 'landing',
        messages: [],
        input: '',
        loading: false,
        priorities: [],
        prioritiesSubmitted: false,
        choices: [],
        choicesTitle: '',
        uncertainties: [],
        dragIndex: null,
        examples: [],
        newPriority: '',

        allExamples: [
            { emoji: '💼', label: 'Should I take this job?', text: 'Should I take this job offer?' },
            { emoji: '💼', label: 'Should I ask for a raise?', text: 'Should I ask my boss for a raise?' },
            { emoji: '💼', label: 'Should I quit my job?', text: 'Should I quit my current job?' },
            { emoji: '💼', label: 'Should I switch careers?', text: 'Should I switch to a different career?' },
            { emoji: '🏠', label: 'Should I move?', text: 'Should I move to a new city?' },
            { emoji: '🏠', label: 'Should I buy or rent?', text: 'Should I buy a place or keep renting?' },
            { emoji: '🏠', label: 'Should I get a roommate?', text: 'Should I get a roommate to save money?' },
            { emoji: '❤️', label: 'Should I text them back?', text: 'Should I text them back?' },
            { emoji: '❤️', label: 'Should I go on another date?', text: 'Should I go on another date with them?' },
            { emoji: '❤️', label: 'Should I end this?', text: 'Should I end this relationship?' },
            { emoji: '❤️', label: 'Should I say I love you?', text: 'Should I tell them I love them?' },
            { emoji: '✈️', label: 'Where should I travel?', text: 'Where should I go on my next trip?' },
            { emoji: '✈️', label: 'Should I book the trip?', text: 'Should I book this trip?' },
            { emoji: '🎓', label: 'Should I go back to school?', text: 'Should I go back to school?' },
            { emoji: '💰', label: 'Should I buy this?', text: 'Should I make this big purchase?' },
            { emoji: '🎯', label: 'Should I start this project?', text: 'Should I start this side project?' },
        ],

        init() {
            this.examples = this.allExamples
                .sort(() => Math.random() - 0.5)
                .slice(0, 4);
        },

        useExample(text) {
            this.input = text;
        },

        async submitInitial() {
            if (!this.input.trim() || this.loading) return;
            this.messages.push({ role: 'user', content: this.input });
            this.view = 'chat';
            this.input = '';
            this.loading = true;
            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ messages: this.messages }),
                });
                const data = await res.json();
                this.messages.push({ role: 'assistant', content: data.response });
            } catch (e) {
                this.messages.push({ role: 'assistant', content: 'Error: ' + e.message });
            }
            this.loading = false;
        },

        async sendMessage() {
            if (!this.input.trim() || this.loading) return;
            this.messages.push({ role: 'user', content: this.input });
            this.input = '';
            this.loading = true;
            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ messages: this.messages }),
                });
                const data = await res.json();
                this.messages.push({ role: 'assistant', content: data.response });
            } catch (e) {
                this.messages.push({ role: 'assistant', content: 'Error: ' + e.message });
            }
            this.loading = false;
        },

        async getPriorities() {
            this.loading = true;
            try {
                const res = await fetch('/api/priorities', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ messages: this.messages }),
                });
                const data = await res.json();
                this.priorities = data.priorities;
                this.prioritiesSubmitted = false;
            } catch (e) {
                throw new Error('Error: ' + e.message);
            }
            this.loading = false;
        },

        async submitPriorities() {
            this.prioritiesSubmitted = true;
            this.messages.push({
                role: 'user',
                content: 'My priorities (ranked): ' + this.priorities.join(', '),
            });
            this.loading = true;
            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ messages: this.messages }),
                });
                const data = await res.json();
                this.messages.push({ role: 'assistant', content: data.response });
            } catch (e) {
                this.messages.push({ role: 'assistant', content: 'Error: ' + e.message });
            }
            this.loading = false;
        },

        addPriority() {
            if (this.newPriority.trim()) {
                this.priorities.push(this.newPriority.trim());
                this.newPriority = '';
            }
        },

        async getChoices() {
            this.loading = true;
            try {
                const res = await fetch('/api/choices', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        messages: this.messages,
                        priorities: this.prioritiesSubmitted ? this.priorities : [],
                    }),
                });
                const data = await res.json();
                this.choices = data.choices;
                this.choicesTitle = data.title;
                this.uncertainties = data.uncertainties;
            } catch (e) {
                throw new Error('Error: ' + e.message);
            }
            this.loading = false;
        },

        reset() {
            this.view = 'landing';
            this.messages = [];
            this.input = '';
            this.priorities = [];
            this.prioritiesSubmitted = false;
            this.newPriority = '';
            this.choices = [];
            this.choicesTitle = '';
            this.uncertainties = [];
        },

        dragStart(i) {
            this.dragIndex = i;
        },

        dragOver(e) {
            e.target.closest('li')?.classList.add('drag-over');
        },

        dragLeave(e) {
            e.target.closest('li')?.classList.remove('drag-over');
        },

        drop(e, i) {
            e.target.closest('li')?.classList.remove('drag-over');
            const item = this.priorities.splice(this.dragIndex, 1)[0];
            this.priorities.splice(i, 0, item);
            this.dragIndex = null;
        },
    };
}

// Browser-only: register with Alpine.js
if (typeof window !== 'undefined' && typeof Alpine !== 'undefined') {
    document.addEventListener('alpine:init', () => {
        Alpine.data('app', createAppData);
    });
}
