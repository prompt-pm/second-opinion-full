/**
 * Tests for Second Opinion Frontend Application
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { createAppData } from '../frontend/app.js';

describe('createAppData', () => {
    let app;

    beforeEach(() => {
        app = createAppData();
        // Reset fetch mock
        global.fetch = vi.fn();
    });

    describe('initial state', () => {
        it('should initialize with landing view', () => {
            expect(app.view).toBe('landing');
        });

        it('should initialize with empty messages', () => {
            expect(app.messages).toEqual([]);
        });

        it('should initialize with empty input', () => {
            expect(app.input).toBe('');
        });

        it('should initialize loading as false', () => {
            expect(app.loading).toBe(false);
        });

        it('should initialize with empty priorities', () => {
            expect(app.priorities).toEqual([]);
        });

        it('should initialize prioritiesSubmitted as false', () => {
            expect(app.prioritiesSubmitted).toBe(false);
        });

        it('should have allExamples array populated', () => {
            expect(app.allExamples.length).toBeGreaterThan(0);
        });
    });

    describe('init()', () => {
        it('should populate examples with 4 random items', () => {
            app.init();
            expect(app.examples).toHaveLength(4);
        });

        it('should select examples from allExamples', () => {
            app.init();
            app.examples.forEach((example) => {
                expect(app.allExamples).toContainEqual(example);
            });
        });
    });

    describe('useExample()', () => {
        it('should set input to the provided text', () => {
            const exampleText = 'Should I take this job?';
            app.useExample(exampleText);
            expect(app.input).toBe(exampleText);
        });
    });

    describe('submitInitial()', () => {
        it('should not submit if input is empty', async () => {
            app.input = '';
            await app.submitInitial();
            expect(app.messages).toEqual([]);
            expect(app.view).toBe('landing');
        });

        it('should not submit if input is only whitespace', async () => {
            app.input = '   ';
            await app.submitInitial();
            expect(app.messages).toEqual([]);
        });

        it('should not submit if already loading', async () => {
            app.input = 'test';
            app.loading = true;
            await app.submitInitial();
            expect(app.messages).toEqual([]);
        });

        it('should add user message and switch to chat view', async () => {
            global.fetch = vi.fn().mockResolvedValue({
                ok: true,
                json: () => Promise.resolve({ response: 'Assistant response' }),
            });

            app.input = 'Should I move?';
            await app.submitInitial();

            expect(app.messages[0]).toEqual({ role: 'user', content: 'Should I move?' });
            expect(app.view).toBe('chat');
        });

        it('should clear input after submission', async () => {
            global.fetch = vi.fn().mockResolvedValue({
                ok: true,
                json: () => Promise.resolve({ response: 'Response' }),
            });

            app.input = 'Test input';
            await app.submitInitial();

            expect(app.input).toBe('');
        });

        it('should add assistant response to messages', async () => {
            global.fetch = vi.fn().mockResolvedValue({
                ok: true,
                json: () => Promise.resolve({ response: 'This is the assistant response' }),
            });

            app.input = 'Test';
            await app.submitInitial();

            expect(app.messages).toHaveLength(2);
            expect(app.messages[1]).toEqual({
                role: 'assistant',
                content: 'This is the assistant response',
            });
        });

        it('should handle fetch errors gracefully', async () => {
            global.fetch = vi.fn().mockRejectedValue(new Error('Network error'));

            app.input = 'Test';
            await app.submitInitial();

            expect(app.messages[1].content).toContain('Error:');
            expect(app.loading).toBe(false);
        });
    });

    describe('sendMessage()', () => {
        it('should not send if input is empty', async () => {
            app.input = '';
            const initialLength = app.messages.length;
            await app.sendMessage();
            expect(app.messages.length).toBe(initialLength);
        });

        it('should not send if loading', async () => {
            app.input = 'test';
            app.loading = true;
            const initialLength = app.messages.length;
            await app.sendMessage();
            expect(app.messages.length).toBe(initialLength);
        });

        it('should add user message and fetch response', async () => {
            global.fetch = vi.fn().mockResolvedValue({
                ok: true,
                json: () => Promise.resolve({ response: 'Response text' }),
            });

            app.input = 'Follow up question';
            await app.sendMessage();

            expect(app.messages[0]).toEqual({ role: 'user', content: 'Follow up question' });
            expect(app.messages[1]).toEqual({ role: 'assistant', content: 'Response text' });
        });

        it('should call /api/chat with correct payload', async () => {
            global.fetch = vi.fn().mockResolvedValue({
                ok: true,
                json: () => Promise.resolve({ response: 'Response' }),
            });

            app.messages = [{ role: 'user', content: 'Previous message' }];
            app.input = 'New message';
            await app.sendMessage();

            expect(global.fetch).toHaveBeenCalledWith('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    messages: [
                        { role: 'user', content: 'Previous message' },
                        { role: 'user', content: 'New message' },
                    ],
                }),
            });
        });
    });

    describe('getPriorities()', () => {
        it('should fetch priorities from API', async () => {
            global.fetch = vi.fn().mockResolvedValue({
                ok: true,
                json: () => Promise.resolve({ priorities: ['Growth', 'Stability', 'Salary'] }),
            });

            app.messages = [{ role: 'user', content: 'Help me decide' }];
            await app.getPriorities();

            expect(app.priorities).toEqual(['Growth', 'Stability', 'Salary']);
            expect(app.prioritiesSubmitted).toBe(false);
        });

        it('should call /api/priorities endpoint', async () => {
            global.fetch = vi.fn().mockResolvedValue({
                ok: true,
                json: () => Promise.resolve({ priorities: [] }),
            });

            app.messages = [{ role: 'user', content: 'Test' }];
            await app.getPriorities();

            expect(global.fetch).toHaveBeenCalledWith('/api/priorities', expect.any(Object));
        });

        it('should throw error on fetch failure', async () => {
            global.fetch = vi.fn().mockRejectedValue(new Error('API error'));

            app.messages = [{ role: 'user', content: 'Test' }];

            await expect(app.getPriorities()).rejects.toThrow('Error: API error');
        });
    });

    describe('addPriority()', () => {
        it('should add priority to the list', () => {
            app.newPriority = 'New priority';
            app.addPriority();
            expect(app.priorities).toContain('New priority');
        });

        it('should clear newPriority after adding', () => {
            app.newPriority = 'Test';
            app.addPriority();
            expect(app.newPriority).toBe('');
        });

        it('should not add empty priority', () => {
            app.newPriority = '';
            app.addPriority();
            expect(app.priorities).toEqual([]);
        });

        it('should not add whitespace-only priority', () => {
            app.newPriority = '   ';
            app.addPriority();
            expect(app.priorities).toEqual([]);
        });

        it('should trim whitespace from priority', () => {
            app.newPriority = '  Important item  ';
            app.addPriority();
            expect(app.priorities[0]).toBe('Important item');
        });
    });

    describe('submitPriorities()', () => {
        it('should set prioritiesSubmitted to true', async () => {
            global.fetch = vi.fn().mockResolvedValue({
                ok: true,
                json: () => Promise.resolve({ response: 'OK' }),
            });

            app.priorities = ['A', 'B'];
            await app.submitPriorities();

            expect(app.prioritiesSubmitted).toBe(true);
        });

        it('should add priorities message to conversation', async () => {
            global.fetch = vi.fn().mockResolvedValue({
                ok: true,
                json: () => Promise.resolve({ response: 'Response' }),
            });

            app.priorities = ['Growth', 'Stability'];
            await app.submitPriorities();

            expect(app.messages[0].content).toContain('My priorities (ranked): Growth, Stability');
        });
    });

    describe('getChoices()', () => {
        it('should fetch choices from API', async () => {
            const mockChoices = {
                title: 'What should you do?',
                choices: [{ name: 'Option A', best_case: 'Good', worst_case: 'Bad' }],
                uncertainties: ['Will it work?'],
            };

            global.fetch = vi.fn().mockResolvedValue({
                ok: true,
                json: () => Promise.resolve(mockChoices),
            });

            app.messages = [{ role: 'user', content: 'Help' }];
            await app.getChoices();

            expect(app.choices).toEqual(mockChoices.choices);
            expect(app.choicesTitle).toBe(mockChoices.title);
            expect(app.uncertainties).toEqual(mockChoices.uncertainties);
        });

        it('should include priorities if submitted', async () => {
            global.fetch = vi.fn().mockResolvedValue({
                ok: true,
                json: () => Promise.resolve({ choices: [], title: '', uncertainties: [] }),
            });

            app.messages = [{ role: 'user', content: 'Test' }];
            app.priorities = ['A', 'B'];
            app.prioritiesSubmitted = true;

            await app.getChoices();

            const fetchCall = global.fetch.mock.calls[0];
            const body = JSON.parse(fetchCall[1].body);
            expect(body.priorities).toEqual(['A', 'B']);
        });

        it('should not include priorities if not submitted', async () => {
            global.fetch = vi.fn().mockResolvedValue({
                ok: true,
                json: () => Promise.resolve({ choices: [], title: '', uncertainties: [] }),
            });

            app.messages = [{ role: 'user', content: 'Test' }];
            app.priorities = ['A', 'B'];
            app.prioritiesSubmitted = false;

            await app.getChoices();

            const fetchCall = global.fetch.mock.calls[0];
            const body = JSON.parse(fetchCall[1].body);
            expect(body.priorities).toEqual([]);
        });
    });

    describe('reset()', () => {
        it('should reset all state to initial values', () => {
            // Set some values
            app.view = 'chat';
            app.messages = [{ role: 'user', content: 'test' }];
            app.input = 'some input';
            app.priorities = ['A', 'B'];
            app.prioritiesSubmitted = true;
            app.newPriority = 'new';
            app.choices = [{ name: 'Choice' }];
            app.choicesTitle = 'Title';
            app.uncertainties = ['Question?'];

            app.reset();

            expect(app.view).toBe('landing');
            expect(app.messages).toEqual([]);
            expect(app.input).toBe('');
            expect(app.priorities).toEqual([]);
            expect(app.prioritiesSubmitted).toBe(false);
            expect(app.newPriority).toBe('');
            expect(app.choices).toEqual([]);
            expect(app.choicesTitle).toBe('');
            expect(app.uncertainties).toEqual([]);
        });
    });

    describe('drag and drop', () => {
        it('dragStart should set dragIndex', () => {
            app.dragStart(2);
            expect(app.dragIndex).toBe(2);
        });

        it('drop should reorder priorities', () => {
            app.priorities = ['A', 'B', 'C', 'D'];
            app.dragIndex = 0;

            // Mock event with closest method
            const mockEvent = {
                target: {
                    closest: () => ({
                        classList: { remove: vi.fn() },
                    }),
                },
            };

            app.drop(mockEvent, 2);

            // 'A' was at index 0, dropped at index 2
            expect(app.priorities).toEqual(['B', 'C', 'A', 'D']);
            expect(app.dragIndex).toBe(null);
        });
    });
});
