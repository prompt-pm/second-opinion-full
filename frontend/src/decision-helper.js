import { marked } from 'marked';

// Configure marked to handle line breaks properly
marked.setOptions({
  breaks: true,  // Convert line breaks to <br>
  gfm: true,     // Enable GitHub Flavored Markdown
});

export function decisionHelper() {
    return {
        situation: '',
        isLoading: false,
        results: null,
        selectedExample: '',
        aiChoice: null,
        selectedOptionIndex: null,
        currentView: 'input', // Can be 'input' or 'conversation'
        error: '',
        showToast: false,
        toastMessage: '',
        toastIcon: 'info', // Add this new property to track the toast icon type
        showUndoButton: false,
        _toastTimeout: null, // For tracking the toast timeout
        darkMode: localStorage.getItem('darkMode') === 'true' || false,
        messages: [], // Array to store conversation messages
        userMessage: '', // For the input box at the bottom
        latestChoicesIndex: -1, // Track the index of the most recent choices message
        conversationHistory: [], // Array to store conversation history
        showHistoryModal: false, // Controls visibility of history modal
        currentConversationId: null, // ID of the current conversation
        currentChoices: [], // Track the currently visible choices (for removal feature)
        activeButtonLoading: null, // Track which button is currently loading
        isProcessingAction: false, // Controls visibility of CTAs during action processing
        sharingDecision: false, // Track when sharing is in progress
        showVideoModal: false, // Controls visibility of the video modal
        newPriority: '',
        latestPrioritiesIndex: -1,
        // Mobile drag state for priorities
        priorityDraggedIndex: null,
        priorityTouchStartY: null,
        priorityTouchStartX: null,
        priorityTouchElement: null,
        priorityLongPressTimer: null,
        priorityIsDragging: false,
        priorityCurrentTouchTarget: null,
        priorityDragClone: null,
        priorityTouchOffsetX: 0,
        priorityTouchOffsetY: 0,
        priorityOriginalPosition: null,
        priorityPlaceholderElement: null,
        selectedAdvisor: 'thought_partner',
        customAdvisor: '',
        
        baseUrl: (() => {
            if (window.location.hostname === '127.0.0.1' || window.location.hostname === 'localhost') {
                return 'http://127.0.0.1:7000';
            } else {
                return 'https://prompt-pm--values-fastapi-app.modal.run';
            }
        })(),
        
        examples: {
            newJob: {
                situation: "I'm considering a new role at work."
            },
            moveCity: {
                situation: "I'm thinking about a new place to live."
            },
            vacation: {
                situation: "I'm thinking about a new vacation spot."
            },
            school: {
                situation: "I'm considering going back to school."
            },
            date: {
                situation: "I'm considering going on a new date."
            }
        },

        init() {
            // Check for dark mode preference
            const savedDarkMode = localStorage.getItem('darkMode');
            
            if (savedDarkMode === null) {
                // If no preference is saved, check system preference
                const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
                this.darkMode = prefersDark;
                localStorage.setItem('darkMode', prefersDark);
            } else {
                this.darkMode = savedDarkMode === 'true';
            }
            
            // Apply dark mode class if needed
            if (this.darkMode) {
                document.documentElement.classList.add('dark');
            } else {
                document.documentElement.classList.remove('dark');
            }

            // Load conversation history from localStorage
            this.loadConversationHistory();

            // If URL path is /decision/:id, load shared decision and stop init
            const sharedMatch = window.location.pathname.match(/^\/decision\/(.+)$/);
            if (sharedMatch && sharedMatch[1]) {
                this.loadSharedDecision(sharedMatch[1]);
                return;
            }

            // Restore current view state from localStorage
            const savedView = localStorage.getItem('currentView');
            if (savedView) {
                // Only restore 'conversation' view if there are messages
                if (savedView === 'conversation' && this.messages.length > 0) {
                    this.currentView = 'conversation';
                }
            }

            // Listen for system dark mode changes
            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
                if (localStorage.getItem('darkMode') === null) {
                    this.darkMode = e.matches;
                    localStorage.setItem('darkMode', e.matches);
                    
                    if (e.matches) {
                        document.documentElement.classList.add('dark');
                    } else {
                        document.documentElement.classList.remove('dark');
                    }
                }
            });

            // Set up visualViewport listener for keyboard events
            if (window.visualViewport) {
                window.visualViewport.addEventListener('resize', () => {
                    // If we're in the conversation view and the keyboard is open
                    if (this.currentView === 'conversation') {
                        const viewportHeight = window.innerHeight;
                        const keyboardHeight = viewportHeight - window.visualViewport.height;
                        
                        // If keyboard is opening (height > 50px)
                        if (keyboardHeight > 50) {
                            // Add a small delay to let the keyboard fully open
                            setTimeout(() => {
                                // Find the input element
                                const inputElement = document.getElementById('message-input');
                                if (inputElement) {
                                    // Scroll to make sure the input is visible
                                    inputElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
                                }
                            }, 300);
                        }
                    }
                });
            }

            // Focus the situation input
            this.$nextTick(() => {
                const textarea = document.getElementById('situation');
                if (textarea) {
                    textarea.focus();
                }
            });
        },

        loadConversationHistory() {
            try {
                const savedHistory = localStorage.getItem('conversationHistory');
                if (savedHistory) {
                    this.conversationHistory = JSON.parse(savedHistory);
                    // Sort by timestamp, most recent first
                    this.conversationHistory.sort((a, b) => b.timestamp - a.timestamp);
                }
            } catch (error) {
                console.error('Error loading conversation history:', error);
                this.conversationHistory = [];
            }
        },

        saveConversationHistory() {
            try {
                localStorage.setItem('conversationHistory', JSON.stringify(this.conversationHistory));
            } catch (error) {
                console.error('Error saving conversation history:', error);
                this.showToast = true;
                this.toastMessage = 'Error saving conversation history';
                setTimeout(() => {
                    this.showToast = false;
                }, 3000);
            }
        },

        saveCurrentConversation() {
            // Don't save empty conversations
            if (!this.messages.length) return;
            
            // Generate a title from the first user message
            const firstUserMessage = this.messages.find(m => m.role === 'user');
            const title = firstUserMessage ? 
                (firstUserMessage.content.length > 50 ? 
                    firstUserMessage.content.substring(0, 50) + '...' : 
                    firstUserMessage.content) : 
                'Untitled Decision';
            
            // Create a conversation object
            const conversation = {
                id: this.currentConversationId || this.generateId(),
                title: title,
                messages: this.messages,
                situation: this.situation,
                timestamp: Date.now(),
                latestChoicesIndex: this.latestChoicesIndex,
                selectedOptionIndex: this.selectedOptionIndex,
                results: this.results
            };
            
            // Set the current conversation ID
            this.currentConversationId = conversation.id;
            
            // Check if this conversation already exists in history
            const existingIndex = this.conversationHistory.findIndex(c => c.id === conversation.id);
            
            if (existingIndex !== -1) {
                // Update existing conversation
                this.conversationHistory[existingIndex] = conversation;
            } else {
                // Add new conversation to history
                this.conversationHistory.unshift(conversation);
            }
            
            // Save to localStorage
            this.saveConversationHistory();
        },
        
        loadConversation(id) {
            const conversation = this.conversationHistory.find(c => c.id === id);
            if (!conversation) return;
            
            // Load conversation data
            this.messages = JSON.parse(JSON.stringify(conversation.messages));
            this.results = conversation.results ? JSON.parse(JSON.stringify(conversation.results)) : null;
            this.selectedOptionIndex = conversation.selectedOptionIndex;
            this.currentConversationId = conversation.id;
            this.currentView = 'conversation';
            localStorage.setItem('currentView', 'conversation');
            
            // Find the latest choices message in the loaded conversation
            this.latestChoicesIndex = -1;
            for (let i = this.messages.length - 1; i >= 0; i--) {
                if (this.messages[i].type === 'choices') {
                    this.latestChoicesIndex = i;
                    this.results = this.messages[i].content;
                    break;
                }
            }
            
            // Find the latest priorities message in the loaded conversation
            this.latestPrioritiesIndex = -1;
            for (let i = this.messages.length - 1; i >= 0; i--) {
                if (this.messages[i].type === 'priorities') {
                    this.latestPrioritiesIndex = i;
                    break;
                }
            }
            
            // Initialize the current choices from results
            this.initializeCurrentChoices();
            
            // Format markdown in assistant messages
            this.messages.forEach(message => {
                if (message.role === 'assistant' && !message.type && typeof message.content === 'string') {
                    message.content = marked.parse(message.content);
                }
            });
            
            // Close the history modal
            this.showHistoryModal = false;
            
            // Scroll to bottom
            this.$nextTick(() => {
                this.scrollToMessage();
            });
        },

        deleteConversation(id) {
            // Remove the conversation from history
            this.conversationHistory = this.conversationHistory.filter(c => c.id !== id);
            
            // Save updated history
            this.saveConversationHistory();
            
            // If we deleted the current conversation, reset the form
            if (this.currentConversationId === id) {
                this.resetForm();
            }
        },

        startNewDecision() {
            // Save current conversation before starting a new one
            this.saveCurrentConversation();
            
            // Reset the form
            this.resetForm();
        },

        openHistoryModal() {
            // Save current conversation before opening history
            this.saveCurrentConversation();
            
            // Show the history modal
            this.showHistoryModal = true;
        },

        generateId() {
            return Date.now().toString(36) + Math.random().toString(36).substr(2, 5);
        },

        formatDate(timestamp) {
            const date = new Date(timestamp);
            const now = new Date();
            const isToday = date.toDateString() === now.toDateString();
            
            // Create a new date for yesterday without modifying 'now'
            const yesterday = new Date(now);
            yesterday.setDate(yesterday.getDate() - 1);
            const isYesterday = yesterday.toDateString() === date.toDateString();
            
            // Format time as "1:30 PM"
            const timeString = date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
            
            if (isToday) {
                return `Today ${timeString}`;
            } else if (isYesterday) {
                return `Yesterday ${timeString}`;
            } else {
                // Format date as "Feb 25"
                const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
                const month = monthNames[date.getMonth()];
                const day = date.getDate();
                return `${month} ${day} ${timeString}`;
            }
        },

        toggleDarkMode() {
            this.darkMode = !this.darkMode;
            localStorage.setItem('darkMode', this.darkMode);
            
            if (this.darkMode) {
                document.documentElement.classList.add('dark');
            } else {
                document.documentElement.classList.remove('dark');
            }
        },

        fillExampleData() {
            if (this.selectedExample in this.examples) {
                this.situation = this.examples[this.selectedExample].situation;
                // Add this timeout to allow the model to update before adjusting height
                setTimeout(() => {
                    const textarea = document.getElementById('situation');
                    if (textarea) {
                        textarea.style.height = '';
                        textarea.style.height = Math.max(textarea.scrollHeight, 56) + 'px';
                    }
                }, 0);
            } else {
                this.situation = '';
            }
        },

        resetForm() {
            this.situation = '';
            this.currentView = 'input';
            localStorage.setItem('currentView', 'input');
            this.results = null;
            this.aiChoice = null;
            this.selectedOptionIndex = null;
            this.messages = [];
            this.userMessage = '';
            this.currentConversationId = null; // Clear current conversation ID
            
            // Reset textarea height
            const textarea = document.getElementById('situation');
            if (textarea) {
                textarea.style.height = '56px'; // Reset to minimum height
            }
        },

        /**
         * Improved scroll function that scrolls to show the latest message while keeping context
         * @param {number} [offset=100] - Optional offset from the bottom of the message
         */
        scrollToMessage(offset = 100) {
            this.$nextTick(() => {
                // Find the message container
                const messageContainer = document.querySelector('.overflow-y-auto');
                if (!messageContainer) return;
                
                // Find the last message element
                const messages = messageContainer.querySelectorAll('.animate__fadeIn');
                if (messages.length === 0) return;
                
                const lastMessage = messages[messages.length - 1];
                
                // Calculate position to scroll to
                const rect = lastMessage.getBoundingClientRect();
                
                // Get viewport height
                const viewportHeight = window.innerHeight;
                
                // Calculate how much of the message is visible
                const visibleHeight = Math.min(
                    rect.bottom,
                    viewportHeight
                ) - Math.max(rect.top, 0);
                
                // If less than 70% of the message is visible, scroll to make it fully visible
                if (visibleHeight < rect.height * 0.7) {
                    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
                    
                    // Calculate a position that ensures the message is visible with some context above
                    // We want the top of the message to be 'offset' pixels from the top of the viewport
                    const targetPosition = scrollTop + rect.top - offset;
                    
                    // Get keyboard height if visualViewport API is available
                    let keyboardHeight = 0;
                    if (window.visualViewport) {
                        keyboardHeight = viewportHeight - window.visualViewport.height;
                    }
                    
                    // Scroll with smooth behavior
                    window.scrollTo({
                        top: targetPosition,
                        behavior: 'smooth'
                    });
                    
                    // If keyboard is open, add extra scroll after a delay
                    if (keyboardHeight > 50) {
                        setTimeout(() => {
                            window.scrollTo({
                                top: targetPosition + keyboardHeight,
                                behavior: 'smooth'
                            });
                        }, 300);
                    }
                }
            });
        },

        sendMessage(isInitialSubmission = false) {
            // For initial submission, use situation; for follow-ups, use userMessage
            const messageContent = isInitialSubmission ? this.situation : this.userMessage;
            
            if (!messageContent.trim() || this.isLoading) return;
            
            // Add user message to conversation
            this.messages.push({
                role: 'user',
                content: messageContent,
                isEditing: false,
                editedContent: messageContent
            });
            
            // Clear input field for follow-up messages
            if (!isInitialSubmission) {
                this.userMessage = '';
            } else {
                // Initial submission specific actions
                this.results = null;
                this.aiChoice = null;
                this.currentView = 'conversation'; // Change to conversation view
                localStorage.setItem('currentView', 'conversation');
                
                // Trigger Google Ads conversion tracking for initial submissions
                if (typeof gtag_report_conversion === 'function') {
                    gtag_report_conversion();
                }
            }
            
            this.isLoading = true;
            this.error = '';

            // Process messages to ensure content is stringified if it's an object
            const processedMessages = this.messages.map(msg => ({
                ...msg,
                content: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)
            }));

            fetch(`${this.baseUrl}/api/query/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    messages: processedMessages, // Use processed messages
                    advisor: this.selectedAdvisor,
                    custom_prefix: this.selectedAdvisor === 'custom' ? this.customAdvisor : null
                }),
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                const responseType = data.response_type;
                const payload = data.response;
                
                if (responseType === 'choices') {
                    this.results = payload;
                    this.messages.push({ role: 'assistant', type: 'choices', content: payload, choices: payload });
                    this.latestChoicesIndex = this.messages.length - 1;
                    this.initializeCurrentChoices();
                    this.scrollToMessage();
                } else if (responseType === 'message') {
                    let formattedText = payload.text;
                    formattedText = marked.parse(formattedText);
                    if (payload.citations) {
                        payload.citations.forEach((citation, i) => {
                            const tag = `[${i+1}]`;
                            formattedText = formattedText.replaceAll(tag, `<a href="${citation}" target="_blank" class="text-sky-500 hover:text-sky-700">${tag}</a>`);
                        });
                    }
                    this.messages.push({ role: 'assistant', content: formattedText, suggested_messages: payload.suggested_messages || [] });
                    this.scrollToMessage();
                } else if (responseType === 'priorities') {
                    this.messages.push({ role: 'assistant', type: 'priorities', content: payload.objectives });
                    this.latestPrioritiesIndex = this.messages.length - 1;
                    this.scrollToMessage();
                } else if (responseType === 'objections') {
                    this.messages.push({ role: 'assistant', type: 'objections', content: payload });
                    this.scrollToMessage();
                } else {
                    this.messages.push({ role: 'assistant', content: "Sorry, I couldn't process your request.", suggested_messages: [] });
                    this.scrollToMessage();
                }
            })
            .catch(error => {
                console.error('Error details:', error);
                this.error = 'We couldn\'t submit your message, please try again.';
                this.showToast = true;
                this.toastMessage = this.error;
                this.toastIcon = 'error'; // Set the error icon
                setTimeout(() => {
                    this.showToast = false;
                }, 3000);
                

            })
            .finally(() => {
                this.isLoading = false;
                
                // Save the conversation to history
                this.saveCurrentConversation();

                // After the first message navigate to the conversation path
                if (isInitialSubmission && this.currentConversationId) {
                    const path = `/conversation/${this.currentConversationId}`;
                    history.pushState({ path, timestamp: Date.now(), conversationId: this.currentConversationId }, '', path);
                }
                
                // Focus on the message input at the bottom
                this.$nextTick(() => {
                    const messageInput = document.getElementById('message-input');
                    if (messageInput) {
                        messageInput.focus();
                    }
                });
            });
        },

        // Alias methods for backward compatibility and clarity
        submitChoices() {
            this.sendMessage(true);
        },

        sendFollowUpMessage() {
            if (!this.userMessage.trim() || this.isLoading) return;
            
            // Add user message to conversation
            this.messages.push({
                role: 'user',
                content: this.userMessage,
                isEditing: false,
                editedContent: this.userMessage
            });
            
            // Clear input field
            this.userMessage = '';
            
            // Scroll to show the user's message immediately
            this.scrollToMessage(150);
            
            // Then proceed with the API call
            this.isLoading = true;
            this.error = '';

            // Process messages to ensure content is stringified if it's an object
            const processedMessages = this.messages.map(msg => ({
                ...msg,
                content: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)
            }));

            fetch(`${this.baseUrl}/api/query/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    messages: processedMessages, // Use processed messages
                    advisor: this.selectedAdvisor,
                    custom_prefix: this.selectedAdvisor === 'custom' ? this.customAdvisor : null
                }),
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                const responseType = data.response_type;
                const payload = data.response;
                
                if (responseType === 'choices') {
                    this.results = payload;
                    this.messages.push({ role: 'assistant', type: 'choices', content: payload, choices: payload });
                    this.latestChoicesIndex = this.messages.length - 1;
                    this.initializeCurrentChoices();
                    this.scrollToMessage();
                } else if (responseType === 'message') {
                    let formattedText = payload.text;
                    formattedText = marked.parse(formattedText);
                    if (payload.citations) {
                        payload.citations.forEach((citation, i) => {
                            const tag = `[${i+1}]`;
                            formattedText = formattedText.replaceAll(tag, `<a href="${citation}" target="_blank" class="text-sky-500 hover:text-sky-700">${tag}</a>`);
                        });
                    }
                    this.messages.push({ role: 'assistant', content: formattedText, suggested_messages: payload.suggested_messages || [] });
                    this.scrollToMessage();
                } else if (responseType === 'priorities') {
                    this.messages.push({ role: 'assistant', type: 'priorities', content: payload.objectives });
                    this.latestPrioritiesIndex = this.messages.length - 1;
                    this.scrollToMessage();
                } else if (responseType === 'objections') {
                    this.messages.push({ role: 'assistant', type: 'objections', content: payload });
                    this.scrollToMessage();
                } else {
                    this.messages.push({ role: 'assistant', content: "Sorry, I couldn't process your request.", suggested_messages: [] });
                    this.scrollToMessage();
                }
            })
            .catch(error => {
                console.error('Error details:', error);
                this.error = 'We couldn\'t submit your message, please try again.';
                this.showToast = true;
                this.toastMessage = this.error;
                this.toastIcon = 'error'; // Set the error icon
                setTimeout(() => {
                    this.showToast = false;
                }, 3000);
            })
            .finally(() => {
                this.isLoading = false;
                
                // Save the conversation to history
                this.saveCurrentConversation();
                
                // Focus on the message input at the bottom
                this.$nextTick(() => {
                    const messageInput = document.getElementById('message-input');
                    if (messageInput) {
                        messageInput.focus();
                    }
                });
            });
        },

        addAlternative() {
            if (!this.results || !this.results.choices || this.isLoading) return;
            
            this.isLoading = true;
            this.isProcessingAction = true; // Set flag to hide CTAs
            this.activeButtonLoading = 'add';
            this.error = '';

            fetch(`${this.baseUrl}/api/add_alternative/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    results: this.results
                }),
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.new_alternative) {
                    // Add the new alternative to the results object
                    this.results.choices.push(data.new_alternative);
                    
                    // Find and update the choices message in the messages array
                    if (this.latestChoicesIndex !== -1) {
                        // Create a new copy of the choices to ensure reactivity
                        const updatedChoices = {...this.messages[this.latestChoicesIndex].choices};
                        updatedChoices.choices = [...updatedChoices.choices, data.new_alternative];
                        this.messages[this.latestChoicesIndex].choices = updatedChoices;
                        
                        // Also add to currentChoices for the UI
                        this.currentChoices.push(data.new_alternative);
                    }
                    
                    // Show notification
                    this.showToast = true;
                    this.toastMessage = "Added a new option";
                    setTimeout(() => {
                        this.showToast = false;
                    }, 3000);
                    
                    // Save the conversation with the new option
                    this.saveCurrentConversation();
                } else {
                    this.error = 'Failed to generate a new alternative.';
                    this.showToast = true;
                    this.toastMessage = this.error;
                    this.toastIcon = 'error'; // Set the error icon
                    setTimeout(() => {
                        this.showToast = false;
                    }, 3000);
                }
            })
            .catch(error => {
                this.error = 'An error occurred while fetching data. Please try again.';
                this.showToast = true;
                this.toastMessage = this.error;
                this.toastIcon = 'error'; // Set the error icon
                setTimeout(() => {
                    this.showToast = false;
                }, 3000);
                console.error('Error:', error);
            })
            .finally(() => {
                this.isLoading = false;
                this.isProcessingAction = false; // Reset flag to show CTAs on the latest message
                this.activeButtonLoading = null;
            });
        },

        chooseForMe() {
            if (!this.results || !this.results.choices || this.isLoading) return;
            
            this.isLoading = true;
            this.isProcessingAction = true; // Set flag to hide CTAs
            this.activeButtonLoading = 'choose';
            this.aiChoice = null;
            
            fetch(`${this.baseUrl}/api/choose/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    situation: this.messages.length > 0 ? this.messages[0].content : this.situation,
                    results: this.results,
                    selectedIndex: this.selectedOptionIndex !== null ? this.selectedOptionIndex : undefined
                }),
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                this.aiChoice = data;
                this.selectedOptionIndex = data.chosen_index;
                
                // Update the choices in the results object
                if (this.results.choices[data.chosen_index]) {
                    this.results.choices[data.chosen_index].explanation = data.explanation;
                }
                
                // Update currentChoices array (if it exists)
                if (this.currentChoices && this.currentChoices.length > 0 && 
                    data.chosen_index < this.currentChoices.length) {
                    this.currentChoices[data.chosen_index].explanation = data.explanation;
                }
                
                // Update the choices message to show the selected option with explanation
                const choicesMessageIndex = this.latestChoicesIndex;
                if (choicesMessageIndex !== -1) {
                    // Create a new copy of the choices to ensure reactivity
                    const updatedChoices = {...this.messages[choicesMessageIndex].choices};
                    if (updatedChoices.choices[data.chosen_index]) {
                        updatedChoices.choices[data.chosen_index].explanation = data.explanation;
                        this.messages[choicesMessageIndex].choices = updatedChoices;
                    }
                }
                
                // Show a toast notification
                this.showToastMessage(`Selected option: ${this.results.choices[data.chosen_index].name}`);
                
                // Use improved scroll function
                this.scrollToMessage();
                
                // Save the conversation with the recommendation
                this.saveCurrentConversation();
            })
            .catch(error => {
                this.error = 'An error occurred while getting AI recommendation. Please try again.';
                this.showToast = true;
                this.toastMessage = this.error;
                this.toastIcon = 'error'; // Set the error icon
                setTimeout(() => {
                    this.showToast = false;
                }, 3000);
                console.error('Error:', error);
            })
            .finally(() => {
                this.isLoading = false;
                this.isProcessingAction = false; // Reset flag to show CTAs on the latest message
                this.activeButtonLoading = null;
            });
        },

        generateNextSteps() {
            this.isLoading = true;
            this.isProcessingAction = true; // Set flag to hide CTAs
            this.activeButtonLoading = 'nextsteps';
            
            // Create payload for API request
            const payload = {
                situation: this.messages.length > 0 ? this.messages[0].content : this.situation,
                results: this.results || {}
            };
            
            // Check if we have a selected option - if so, add it to payload for post-decision steps
            // Note: We'll still work if no option is selected
            if (this.selectedOptionIndex !== null && this.results && this.results.choices) {
                const selectedChoice = this.results.choices[this.selectedOptionIndex];
                payload.choice_name = selectedChoice.name;
                payload.choice_index = this.selectedOptionIndex;
                if (this.results.id) {
                    payload.decision_id = this.results.id;
                }
            }
            
            fetch(`${this.baseUrl}/api/next_steps`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload),
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                if (data.next_steps && data.next_steps.length > 0) {
                    // Check if there's already a next_steps message
                    const existingNextStepsIndex = this.messages.findIndex(msg => msg.type === 'next_steps');
                    
                    // Create title based on whether we're in pre-decision or post-decision mode
                    const title = this.selectedOptionIndex !== null && this.results && this.results.choices
                        ? `Next Steps for "${this.results.choices[this.selectedOptionIndex].name}"`
                        : "Next Steps";
                    
                    const newNextStepsMessage = {
                        role: 'assistant',
                        type: 'next_steps',
                        content: {
                            title: title,
                            steps: data.next_steps.map(step => ({
                                text: step
                            }))
                        }
                    };
                    
                    if (existingNextStepsIndex !== -1) {
                        // Replace the existing next_steps message
                        this.messages[existingNextStepsIndex] = newNextStepsMessage;
                    } else {
                        // Add a new message with the next steps
                        this.messages.push(newNextStepsMessage);
                    }
                    
                    // Use improved scroll function
                    this.scrollToMessage();
                    
                    // Save the current conversation
                    this.saveCurrentConversation();
                } else {
                    this.showToastMessage('No action items generated', false, 'error');
                }
            })
            .catch(error => {
                console.error('Error generating next steps:', error);
                this.showToastMessage('Error generating next steps: ' + error.message, false, 'error');
            })
            .finally(() => {
                this.isLoading = false;
                this.isProcessingAction = false; // Reset flag to show CTAs on the latest message
                this.activeButtonLoading = null;
            });
        },
        
        suggestAdditionalAction(message) {
            if (!message || !message.content || !message.content.steps) {
                this.showToastMessage('No next steps found', false, 'error');
                return;
            }
            
            this.isLoading = true;
            
            // Get the situation from the first message or the situation field
            const situation = this.messages.length > 0 ? this.messages[0].content : this.situation;
            
            // Extract existing next steps as plain text array
            const existingNextSteps = message.content.steps.map(step => step.text);
            
            fetch(`${this.baseUrl}/api/suggest_additional_action`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    situation: situation,
                    existing_next_steps: existingNextSteps,
                    results: this.results || {}
                }),
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                if (data.additional_action) {
                    // Add the new action to the steps array
                    message.content.steps.push({
                        text: data.additional_action
                    });
                    
                    // Save the conversation
                    this.saveCurrentConversation();
                    
                    // Scroll to show the updated list
                    this.scrollToMessage();
                    
                    // Show success message
                    this.showToastMessage('Added new action');
                } else {
                    this.showToastMessage('Could not generate a new action', false, 'error');
                }
            })
            .catch(error => {
                console.error('Error suggesting additional action:', error);
                this.showToastMessage('Error suggesting action: ' + error.message, false, 'error');
            })
            .finally(() => {
                this.isLoading = false;
            });
        },

        showToastMessage(message, showUndo = false, icon = 'info') {
            // Clear any existing timeout to prevent premature hiding
            if (this._toastTimeout) {
                clearTimeout(this._toastTimeout);
                this._toastTimeout = null;
            }
            
            this.toastMessage = message;
            this.showUndoButton = showUndo;
            this.toastIcon = icon; // Set the icon type
            this.showToast = true;
            
            this._toastTimeout = setTimeout(() => {
                this.showToast = false;
                this._toastTimeout = null;
            }, 5000);
        },
        
        switchToDecisions() {
            this.currentView = this.messages.length > 0 ? 'conversation' : 'input';
            localStorage.setItem('currentView', this.currentView);
        },
        
        resubmitFromMessage(index) {
            if (index < 0 || index >= this.messages.length || this.isLoading) return;
            
            // Keep messages up to and including the selected message
            const messagesToKeep = this.messages.slice(0, index + 1);
            const selectedMessage = this.messages[index];
            
            // Reset the conversation state
            this.messages = messagesToKeep;
            this.selectedOptionIndex = null;
            this.results = null;
            this.aiChoice = null;
            this.latestChoicesIndex = -1;
            
            // Find the latest choices message if it exists in the kept messages
            for (let i = messagesToKeep.length - 1; i >= 0; i--) {
                if (messagesToKeep[i].type === 'choices') {
                    this.latestChoicesIndex = i;
                    this.results = messagesToKeep[i].content;
                    break;
                }
            }
            
            // If the selected message is from the user, we need to send it again to get a new response
            if (selectedMessage.role === 'user') {
                // Remove the last message (user message) as we'll add it back when sending
                this.messages.pop();
                
                // Set the user message to the content of the selected message
                this.userMessage = selectedMessage.content;
                
                // Send the message
                this.$nextTick(() => {
                    this.sendFollowUpMessage();
                });
            }
            
            // Show a toast notification
            this.showToastMessage(selectedMessage.role === 'user' 
                ? 'Resubmitting message...' 
                : 'Conversation reset to this point');
            
            // Save the updated conversation
            this.saveCurrentConversation();
        },

        editMessage(index) {
            if (index < 0 || index >= this.messages.length || this.isLoading) return;
            
            const selectedMessage = this.messages[index];
            
            // Only allow editing user messages
            if (selectedMessage.role !== 'user') return;
            
            // Set the message as being edited
            selectedMessage.isEditing = true;
            selectedMessage.editedContent = selectedMessage.content;
            
            // Focus the editable content in the next tick
            this.$nextTick(() => {
                const editableElement = document.querySelector(`[data-message-index="${index}"] .editable-message`);
                if (editableElement) {
                    editableElement.focus();
                    // Place cursor at the end
                    const range = document.createRange();
                    const selection = window.getSelection();
                    range.selectNodeContents(editableElement);
                    range.collapse(false);
                    selection.removeAllRanges();
                    selection.addRange(range);
                }
            });
        },
        
        saveEditedMessage(index) {
            if (index < 0 || index >= this.messages.length) return;
            
            const selectedMessage = this.messages[index];
            
            // Only proceed if the message is being edited
            if (!selectedMessage.isEditing) return;
            
            // Get the edited content
            const editedContent = selectedMessage.editedContent.trim();
            
            // Only proceed if the content is not empty and has changed
            if (!editedContent || editedContent === selectedMessage.content) {
                selectedMessage.isEditing = false;
                return;
            }
            
            // Update the message content
            selectedMessage.content = editedContent;
            selectedMessage.isEditing = false;
            
            // Keep messages up to and including the edited message
            const messagesToKeep = this.messages.slice(0, index + 1);
            
            // Reset the conversation state
            this.messages = messagesToKeep;
            this.selectedOptionIndex = null;
            this.results = null;
            this.aiChoice = null;
            this.latestChoicesIndex = -1;
            
            // Find the latest choices message if it exists in the kept messages
            for (let i = messagesToKeep.length - 1; i >= 0; i--) {
                if (messagesToKeep[i].type === 'choices') {
                    this.latestChoicesIndex = i;
                    this.results = messagesToKeep[i].content;
                    break;
                }
            }
            
            // Send the edited message directly without adding a new user message
            this.isLoading = true;
            this.error = '';

            // Process messages to ensure content is stringified if it's an object
            const processedMessages = this.messages.map(msg => ({
                ...msg,
                content: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)
            }));

            fetch(`${this.baseUrl}/api/query/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    messages: processedMessages, // Use processed messages
                    advisor: this.selectedAdvisor,
                    custom_prefix: this.selectedAdvisor === 'custom' ? this.customAdvisor : null
                }),
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                const responseType = data.response_type;
                const payload = data.response;
                
                if (responseType === 'choices') {
                    this.results = payload;
                    this.messages.push({ role: 'assistant', type: 'choices', content: payload, choices: payload });
                    this.latestChoicesIndex = this.messages.length - 1;
                    this.initializeCurrentChoices();
                    this.scrollToMessage();
                } else if (responseType === 'message') {
                    let formattedText = payload.text;
                    formattedText = marked.parse(formattedText);
                    if (payload.citations) {
                        payload.citations.forEach((citation, i) => {
                            const tag = `[${i+1}]`;
                            formattedText = formattedText.replaceAll(tag, `<a href="${citation}" target="_blank" class="text-sky-500 hover:text-sky-700">${tag}</a>`);
                        });
                    }
                    this.messages.push({ role: 'assistant', content: formattedText, suggested_messages: payload.suggested_messages || [] });
                    this.scrollToMessage();
                } else if (responseType === 'priorities') {
                    this.messages.push({ role: 'assistant', type: 'priorities', content: payload.objectives });
                    this.latestPrioritiesIndex = this.messages.length - 1;
                    this.scrollToMessage();
                } else if (responseType === 'objections') {
                    this.messages.push({ role: 'assistant', type: 'objections', content: payload });
                    this.scrollToMessage();
                } else {
                    this.messages.push({ role: 'assistant', content: "Sorry, I couldn't process your request.", suggested_messages: [] });
                    this.scrollToMessage();
                }
            })
            .catch(error => {
                console.error('Error details:', error);
                this.error = 'We couldn\'t submit your message, please try again.';
                this.showToast = true;
                this.toastMessage = this.error;
                this.toastIcon = 'error'; // Set the error icon
                setTimeout(() => {
                    this.showToast = false;
                }, 3000);
            })
            .finally(() => {
                this.isLoading = false;
                
                // Save the conversation to history
                this.saveCurrentConversation();
                
                // Focus on the message input at the bottom
                this.$nextTick(() => {
                    const messageInput = document.getElementById('message-input');
                    if (messageInput) {
                        messageInput.focus();
                    }
                });
            });
        },
        
        cancelEditMessage(index) {
            if (index < 0 || index >= this.messages.length) return;
            
            const selectedMessage = this.messages[index];
            
            // Only proceed if the message is being edited
            if (!selectedMessage.isEditing) return;
            
            // Reset the editing state
            selectedMessage.isEditing = false;
            selectedMessage.editedContent = selectedMessage.content;
        },

        // Generate options based on the entire conversation history
        generateChoices() {
            if (this.isLoading) return;
            
            this.isLoading = true;
            
            // Remove any existing choice messages from the chat history
            if (this.latestChoicesIndex !== -1) {
                this.messages = this.messages.filter(msg => msg.type !== 'choices');
                // Reset the latest choices index
                this.latestChoicesIndex = -1;
                // Reset selection
                this.selectedOptionIndex = null;
            }
            
            // Format messages for API
            const formattedMessages = this.messages.map(msg => ({
                role: msg.role,
                content: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)
            }));
            
            this.error = '';

            fetch(`${this.baseUrl}/api/choices/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message_history: formattedMessages
                }),
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.choices && Array.isArray(data.choices)) {
                    this.results = {
                        title: data.title || "Your options",
                        choices: data.choices,
                        uncertainties: data.uncertainties || [],
                        next_steps: data.next_steps || []
                    };
                    
                    // Add the choices as a special message type in the conversation
                    this.messages.push({
                        role: 'assistant',
                        content: this.results,
                        type: 'choices',
                        choices: this.results
                    });
                    
                    // Update the latest choices index
                    this.latestChoicesIndex = this.messages.length - 1;
                    
                    // Initialize the current choices from results
                    this.initializeCurrentChoices();
                    
                    // Use improved scroll function
                    this.scrollToMessage();
                    
                    // Save the conversation history
                    this.saveCurrentConversation();
                } else {
                    // Handle error
                    this.showToastMessage('Failed to generate options', false, 'error');
                }
            })
            .catch(error => {
                console.error('Error generating options:', error);
                this.showToastMessage('Error generating options: ' + error.message, false, 'error');
            })
            .finally(() => {
                this.isLoading = false;
            });
        },
        
        // Initialize or update the currentChoices array from results
        initializeCurrentChoices() {
            if (this.results && this.results.choices) {
                // Create a deep copy of the choices array to avoid reference issues
                this.currentChoices = JSON.parse(JSON.stringify(this.results.choices));
            } else {
                this.currentChoices = [];
            }
        },
        
        // Remove a choice from the currentChoices array
        removeChoice(index) {
            if (index >= 0 && index < this.currentChoices.length) {
                // If the removed choice was selected, reset selection
                if (this.selectedOptionIndex === index) {
                    this.selectedOptionIndex = null;
                } 
                // If the removed choice was before the selected one, adjust selectedOptionIndex
                else if (this.selectedOptionIndex > index) {
                    this.selectedOptionIndex--;
                }
                
                // Remove the choice at the specified index
                this.currentChoices.splice(index, 1);
                
                // Show toast notification
                this.showToastMessage("Option removed from view");
            }
        },

        // Generate options from any point in the conversation
        generateOptionsFromMessage(messageIndex) {
            if (messageIndex < 0 || messageIndex >= this.messages.length || this.isLoading) return;
            
            // Set both flags to hide CTAs immediately
            this.isProcessingAction = true;
            this.isLoading = true;
            
            // Remove any existing choice messages from the chat history
            if (this.latestChoicesIndex !== -1) {
                this.messages = this.messages.filter(msg => msg.type !== 'choices');
                // Reset the latest choices index
                this.latestChoicesIndex = -1;
                // Reset selection
                this.selectedOptionIndex = null;
            }
            
            // Include conversation context up to this message
            const contextMessages = this.messages.slice(0, messageIndex + 1);
            
            // The backend expects a list of message objects, not a string
            // Convert to the format expected by the backend
            const formattedMessages = contextMessages.map(msg => ({
                role: msg.role,
                content: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)
            }));
            
            // If there are no messages, create a default one
            if (formattedMessages.length === 0) {
                formattedMessages.push({
                    role: 'user',
                    content: 'Help me make a decision'
                });
            }
            
            this.error = '';

            fetch(`${this.baseUrl}/api/choices/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message_history: formattedMessages  // Send as array of message objects
                }),
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.choices && Array.isArray(data.choices)) {
                    this.results = {
                        title: data.title || "Your options",
                        choices: data.choices,
                        uncertainties: data.uncertainties || [],
                        next_steps: data.next_steps || []
                    };
                    
                    // Add the choices as a special message type in the conversation
                    this.messages.push({
                        role: 'assistant',
                        content: this.results,
                        type: 'choices',
                        choices: this.results
                    });
                    
                    // Update the latest choices index
                    this.latestChoicesIndex = this.messages.length - 1;
                    
                    // Use improved scroll function
                    this.scrollToMessage();
                    
                    // Initialize the current choices from results
                    this.initializeCurrentChoices();
                    
                    // Save the conversation history
                    this.saveCurrentConversation();
                } else {
                    // Handle error
                    this.showToastMessage('Failed to generate options', false, 'error');
                }
            })
            .catch(error => {
                console.error('Error generating options:', error);
                this.showToastMessage('Error generating options: ' + error.message, false, 'error');
            })
            .finally(() => {
                this.isLoading = false;
                this.isProcessingAction = false; // Reset processing flag to allow CTAs on the new message
            });
        },

        // Share the current decision 
        shareDecision() {
            if (this.messages.length === 0 || this.sharingDecision) return;
            
            this.sharingDecision = true;
            
            // Process messages to ensure content is stringified if it's an object
            const processedMessages = this.messages.map(msg => ({
                ...msg,
                content: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)
            }));

            fetch(`${this.baseUrl}/api/save_decision/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    message_history: processedMessages // Use processed messages
                }),
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.id) {
                    // Create shareable link using the ID
                    const shareLink = `${window.location.origin}/decision/${data.id}`;
                    
                    // Copy to clipboard
                    navigator.clipboard.writeText(shareLink).then(() => {
                        this.showToastMessage('Link copied to clipboard', false, 'info');
                    }, () => {
                        // Clipboard write failed, just show the link
                        this.showToastMessage('Share link created', false, 'info');
                    });

                    // Create a prompt to show the shareable link
                    const modal = document.createElement('div');
                    modal.className = 'fixed z-50 inset-0 overflow-y-auto flex items-center justify-center bg-black bg-opacity-50';
                    modal.innerHTML = `
                        <div class="bg-white dark:bg-gray-800 rounded-lg p-6 max-w-md w-full mx-4 shadow-xl">
                            <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-4">Share Your Decision</h3>
                            <p class="text-gray-600 dark:text-gray-300 mb-4">Share this link with others to show them your decision:</p>
                            <div class="flex items-center gap-2 mb-6">
                                <input 
                                    type="text" 
                                    value="${shareLink}" 
                                    class="w-full bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-white p-2 rounded-md" 
                                    readonly
                                    onclick="this.select();"
                                />
                                <button 
                                    class="bg-sky-500 hover:bg-sky-600 text-white p-2 rounded-md"
                                    onclick="navigator.clipboard.writeText('${shareLink}').then(() => { 
                                        const copyBtn = document.getElementById('copy-btn');
                                        if (copyBtn) {
                                            copyBtn.innerHTML = '<i class=\\'fas fa-check\\'></i>';
                                            setTimeout(() => { copyBtn.innerHTML = '<i class=\\'fas fa-copy\\'></i>'; }, 2000);
                                        }
                                    })"
                                    id="copy-btn"
                                >
                                    <i class="fas fa-copy"></i>
                                </button>
                            </div>
                            <div class="flex justify-end">
                                <button 
                                    class="bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-900 dark:text-white px-4 py-2 rounded-md"
                                    onclick="this.closest('.fixed').remove();"
                                >
                                    Close
                                </button>
                            </div>
                        </div>
                    `;
                    
                    document.body.appendChild(modal);
                    
                    // Add event listener to close on background click
                    modal.addEventListener('click', (e) => {
                        if (e.target === modal) {
                            modal.remove();
                        }
                    });
                    
                } else {
                    this.showToastMessage('Failed to create share link', false, 'error');
                }
            })
            .catch(error => {
                console.error('Error sharing decision:', error);
                this.showToastMessage('Error creating share link', false, 'error');
            })
            .finally(() => {
                this.sharingDecision = false;
            });
        },

        // Load a shared decision
        loadSharedDecision(decisionId) {
            if (!decisionId) return;
            
            this.isLoading = true;
            this.error = '';
            
            fetch(`${this.baseUrl}/api/get_decision/${decisionId}`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`Decision not found (status: ${response.status})`);
                    }
                    return response.json();
                })
                .then(data => {
                    if (data.message_history && Array.isArray(data.message_history)) {
                        // Reset current state
                        this.resetForm();
                        
                        // Load the shared decision messages
                        this.messages = data.message_history;
                        
                        // Parse priorities list if content is a JSON string
                        this.messages.forEach(message => {
                            if (message.type === 'priorities' && typeof message.content === 'string') {
                                try {
                                    message.content = JSON.parse(message.content);
                                } catch (e) {
                                    console.error('Failed to parse priorities content', e);
                                }
                            }
                        });
                        
                        // Set to conversation view
                        this.currentView = 'conversation';
                        localStorage.setItem('currentView', 'conversation');
                        
                        // Find the latest choices message in the loaded conversation
                        this.latestChoicesIndex = -1;
                        for (let i = this.messages.length - 1; i >= 0; i--) {
                            if (this.messages[i].type === 'choices') {
                                this.latestChoicesIndex = i;
                                this.results = this.messages[i].content;
                                break;
                            }
                        }
                        
                        // Find the latest priorities message in the loaded conversation
                        this.latestPrioritiesIndex = -1;
                        for (let i = this.messages.length - 1; i >= 0; i--) {
                            if (this.messages[i].type === 'priorities') {
                                this.latestPrioritiesIndex = i;
                                break;
                            }
                        }
                        
                        // Initialize the current choices from results
                        this.initializeCurrentChoices();
                        
                        // Format markdown in assistant messages
                        this.messages.forEach(message => {
                            if (message.role === 'assistant' && !message.type && typeof message.content === 'string') {
                                message.content = marked.parse(message.content);
                            }
                        });
                        
                        // Set title from first user message
                        const firstUserMessage = this.messages.find(m => m.role === 'user');
                        const title = firstUserMessage ? 
                            (firstUserMessage.content.length > 50 ? 
                                firstUserMessage.content.substring(0, 50) + '...' : 
                                firstUserMessage.content) : 
                            'Shared Decision';
                        
                        document.title = `Shared: ${title} | say less`;
                        
                        // Show notification
                        this.showToastMessage('Viewing a shared decision', false, 'info');
                    } else {
                        throw new Error('Invalid decision data');
                    }
                })
                .catch(error => {
                    console.error('Error loading shared decision:', error);
                    // If the decision exists in local history, load it; otherwise show error
                    const localEntry = this.conversationHistory.find(c => c.id === decisionId);
                    if (localEntry) {
                        this.loadConversation(decisionId);
                    } else {
                        this.error = error.message || 'Failed to load shared decision';
                        this.showToastMessage(this.error, false, 'error');
                    }
                })
                .finally(() => {
                    this.isLoading = false;
                    
                    // Scroll to show the conversation
                    this.$nextTick(() => {
                        this.scrollToMessage();
                    });
                });
        },

        // Drag and drop handlers for priorities list
        dragStart(event, list) {
            // Store the source index
            const fromIndex = +event.target.closest('li').dataset.index;
            event.dataTransfer.setData('text/plain', fromIndex);
            event.dataTransfer.effectAllowed = 'move';
            // Add placeholder at original position
            const listItem = event.target.closest('li');
            listItem.classList.add('drag-placeholder');
            // Use a custom drag image so the element travels with the cursor
            const rect = listItem.getBoundingClientRect();
            const offsetX = event.clientX - rect.left;
            const offsetY = event.clientY - rect.top;
            event.dataTransfer.setDragImage(listItem, offsetX, offsetY);
        },
        dragEnd(event, list) {
            // Remove placeholder and highlight classes after dragging ends
            document.querySelectorAll('li[data-index]').forEach(li => {
                li.classList.remove('drag-placeholder', 'drag-over');
            });
        },
        dragOver(event) {
            event.preventDefault();
            const li = event.target.closest('li');
            // Highlight the current drop target
            document.querySelectorAll('li[data-index]').forEach(el => el.classList.remove('drag-over'));
            if (li) li.classList.add('drag-over');
        },
        dragLeave(event) {
            const li = event.target.closest('li');
            if (li) li.classList.remove('drag-over');
        },
        drop(event, list) {
            event.preventDefault();
            const li = event.target.closest('li');
            const from = +event.dataTransfer.getData('text/plain');
            const to = li ? +li.dataset.index : null;
            if (to !== null && from !== to) {
                const item = list.splice(from, 1)[0];
                list.splice(to, 0, item);
            }
            // Cleanup placeholder and highlights
            document.querySelectorAll('li[data-index]').forEach(el => {
                el.classList.remove('drag-placeholder', 'drag-over');
            });
        },

        // ----- Mobile touch handlers for priorities -----
        priorityHandleTouchStart(event, index, list) {
            this.priorityTouchStartY = event.touches[0].clientY;
            this.priorityTouchStartX = event.touches[0].clientX;

            this.priorityTouchElement = event.currentTarget.closest('li');
            this.priorityCurrentTouchTarget = index;

            const rect = this.priorityTouchElement.getBoundingClientRect();
            this.priorityTouchOffsetX = this.priorityTouchStartX - rect.left;
            this.priorityTouchOffsetY = this.priorityTouchStartY - rect.top;

            this.priorityOriginalPosition = {
                width: rect.width,
                height: rect.height,
                left: rect.left,
                top: rect.top,
            };

            this.priorityLongPressTimer = setTimeout(() => {
                this.priorityIsDragging = true;
                this.priorityDraggedIndex = index;
                this.createPriorityDragClone();
                this.priorityTouchElement.classList.add('drag-placeholder');
                this.priorityPlaceholderElement = this.priorityTouchElement;
                if (navigator.vibrate) {
                    navigator.vibrate(50);
                }
                this.showToastMessage('Drag to reorder');
            }, 300);
        },

        createPriorityDragClone() {
            this.priorityDragClone = this.priorityTouchElement.cloneNode(true);
            this.priorityDragClone.classList.add('touch-dragging');
            this.priorityDragClone.classList.remove('drag-placeholder');
            this.priorityDragClone.style.position = 'fixed';
            this.priorityDragClone.style.left = `${this.priorityOriginalPosition.left}px`;
            this.priorityDragClone.style.top = `${this.priorityOriginalPosition.top}px`;
            this.priorityDragClone.style.width = `${this.priorityOriginalPosition.width}px`;
            this.priorityDragClone.style.margin = '0';
            this.priorityDragClone.style.zIndex = '9999';
            document.body.appendChild(this.priorityDragClone);
            this.updatePriorityDragClonePosition(this.priorityTouchStartX, this.priorityTouchStartY);
        },

        updatePriorityDragClonePosition(x, y) {
            if (!this.priorityDragClone) return;
            const left = x - this.priorityTouchOffsetX;
            const top = y - this.priorityTouchOffsetY;
            this.priorityDragClone.style.left = `${left}px`;
            this.priorityDragClone.style.top = `${top}px`;
        },

        priorityHandleTouchMove(event) {
            if (!this.priorityIsDragging) {
                const touchY = event.touches[0].clientY;
                const touchX = event.touches[0].clientX;
                if (Math.abs(touchY - this.priorityTouchStartY) > 10 || Math.abs(touchX - this.priorityTouchStartX) > 10) {
                    clearTimeout(this.priorityLongPressTimer);
                }
                return;
            }

            event.preventDefault();
            event.stopPropagation();

            const touchX = event.touches[0].clientX;
            const touchY = event.touches[0].clientY;

            this.updatePriorityDragClonePosition(touchX, touchY);

            const elementsUnderTouch = document.elementsFromPoint(touchX, touchY);
            const targetLi = elementsUnderTouch.find(el =>
                el.tagName === 'LI' &&
                el.dataset.index !== undefined &&
                !el.classList.contains('touch-dragging') &&
                !el.classList.contains('drag-placeholder')
            );

            if (targetLi) {
                const targetIndex = +targetLi.dataset.index;
                if (!Number.isNaN(targetIndex) && targetIndex !== this.priorityDraggedIndex) {
                    document.querySelectorAll('li[data-index]').forEach(li => {
                        if (li !== this.priorityPlaceholderElement) {
                            li.classList.remove('drag-over');
                        }
                    });
                    targetLi.classList.add('drag-over');
                    this.priorityCurrentTouchTarget = targetIndex;
                }
            }
        },

        priorityHandleTouchEnd(event, list) {
            clearTimeout(this.priorityLongPressTimer);

            if (this.priorityIsDragging) {
                if (this.priorityDragClone) {
                    this.priorityDragClone.remove();
                    this.priorityDragClone = null;
                }

                if (this.priorityPlaceholderElement) {
                    this.priorityPlaceholderElement.classList.remove('drag-placeholder');
                    this.priorityPlaceholderElement = null;
                }

                document.querySelectorAll('li[data-index]').forEach(li => li.classList.remove('drag-over'));

                if (this.priorityCurrentTouchTarget !== null && this.priorityCurrentTouchTarget !== this.priorityDraggedIndex) {
                    const item = list.splice(this.priorityDraggedIndex, 1)[0];
                    list.splice(this.priorityCurrentTouchTarget, 0, item);
                }

                this.priorityIsDragging = false;
            }

            this.priorityDraggedIndex = null;
            this.priorityTouchStartY = null;
            this.priorityTouchStartX = null;
            this.priorityTouchElement = null;
            this.priorityCurrentTouchTarget = null;
            this.priorityTouchOffsetX = 0;
            this.priorityTouchOffsetY = 0;
            this.priorityOriginalPosition = null;
        },
        addPriority() {
            if (!this.newPriority.trim()) return;
            this.messages[this.latestPrioritiesIndex].content.push(this.newPriority.trim());
            this.newPriority = '';
        },
        removePriority(index) {
            this.messages[this.latestPrioritiesIndex].content.splice(index, 1);
        },
        updatePriority(index, event) {
            this.messages[this.latestPrioritiesIndex].content[index] = event.target.textContent;
        },
        submitPriorities() {
            const list = this.messages[this.latestPrioritiesIndex].content;
            const text = list.map((p, i) => `${i+1}. ${p}`).join('\n');
            this.messages.splice(this.latestPrioritiesIndex + 1);
            this.userMessage = "My priorities are: " + text;
            this.sendFollowUpMessage();
        },
    };
}