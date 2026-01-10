import './styles.css';
import Alpine from 'alpinejs'
import { decisionHelper } from './decision-helper';
import { actionsHelper } from './actions';
import { marked } from 'marked';

// Browser history navigation functions
// Navigate to a specific path and update the browser's history
function navigateTo(path, state = {}) {
    // Make sure we're using the full path for consistent history management
    console.log('Navigating to:', path);
    const stateObj = { path, timestamp: Date.now(), ...state };
    history.pushState(stateObj, '', path);
}

// Initialize the page based on the URL
function initBrowserNavigation(app) {
    // Handle initial load
    window.addEventListener('load', () => {
        const path = window.location.pathname;
        
        // Set initial state without creating a new history entry
        history.replaceState({ path }, '', path);
        
        // Handle initial page path
        handlePathChange(path, app);
    });
    
    // Handle back/forward navigation
    window.addEventListener('popstate', (event) => {
        console.log('Navigation event:', event.state);
        if (event.state && event.state.path) {
            // We need to make sure we're always accessing the full path
            const fullPath = window.location.pathname;
            handlePathChange(fullPath, app);
        } else {
            // Default to home if no state
            handlePathChange('/', app);
        }
    });
}

// Handle path changes and update app state accordingly
function handlePathChange(path, app) {
    // Extract path segments
    const segments = path.split('/').filter(segment => segment);
    
    if (segments.length === 0 || segments[0] === '') {
        // Home path: "/"
        app._isHandlingHistoryNavigation = true;
        app.resetForm();
        setTimeout(() => {
            app._isHandlingHistoryNavigation = false;
        }, 0);
    /* Disabled actions view
    } else if (segments[0] === 'actions') {
        // Actions path: "/actions"
        app._isHandlingHistoryNavigation = true;
        app.switchToTodos();
        setTimeout(() => {
            app._isHandlingHistoryNavigation = false;
        }, 0);
    */
    } else if (segments[0] === 'conversation' && segments.length > 1) {
        // Conversation path: "/conversation/:id"
        const conversationId = segments[1];
        console.log('Loading conversation from URL:', conversationId);
        
        // Use the loadConversation method but make sure we don't create a new history entry
        const conversation = app.conversationHistory.find(c => c.id === conversationId);
        
        if (conversation) {
            // Set a flag to prevent duplicate history entries
            app._isHandlingHistoryNavigation = true;
            
            // Call the app's loadConversation method directly
            // This will handle loading all the necessary data
            app.loadConversation(conversationId);
            
            // Clear the flag
            setTimeout(() => {
                app._isHandlingHistoryNavigation = false;
            }, 0);
        } else {
            // Conversation not found, go to home
            app.currentView = 'input';
            localStorage.setItem('currentView', 'input');
            history.replaceState({ path: '/' }, '', '/');
        }
    } else if (segments[0] === 'decision' && segments.length > 1) {
        // Shared decision path: "/decision/:id"
        const decisionId = segments[1];
        console.log('Loading shared decision from URL:', decisionId);
        
        // Set a flag to prevent duplicate history entries
        app._isHandlingHistoryNavigation = true;
        
        // Load the shared decision from the server
        app.loadSharedDecision(decisionId);
        
        // Clear the flag
        setTimeout(() => {
            app._isHandlingHistoryNavigation = false;
        }, 0);
    } else if (segments[0] === 'history') {
        // History path: "/history"
        app._isHandlingHistoryNavigation = true;
        app.openHistoryModal();
        setTimeout(() => {
            app._isHandlingHistoryNavigation = false;
        }, 0);
    } else {
        // Default to home for unrecognized paths
        app._isHandlingHistoryNavigation = true;
        app.resetForm();
        setTimeout(() => {
            app._isHandlingHistoryNavigation = false;
        }, 0);
    }
}

// Combine the helpers into a single app
function appData() {
    const decisions = decisionHelper();
    const actions = actionsHelper();
    
    // Create a merged object with all properties and methods
    const merged = {
        ...decisions,
        ...actions,
        _isHandlingHistoryNavigation: false, // Flag to prevent history duplication
        
        // Override init to call both inits
        init() {
            decisions.init.call(this);
            actions.init.call(this);
            
            // Initialize browser navigation
            initBrowserNavigation(this);
        },
        
        // Override navigation methods to use browser history
        switchToTodos() {
            if (!this._isHandlingHistoryNavigation) {
                navigateTo('/actions');
            }
            decisions.switchToTodos.call(this);
        },
        
        switchToDecisions() {
            if (!this._isHandlingHistoryNavigation) {
                navigateTo('/' + (this.messages.length > 0 ? 'conversation/' + this.currentConversationId : ''));
            }
            decisions.switchToDecisions.call(this);
        },
        
        resetForm() {
            if (!this._isHandlingHistoryNavigation) {
                navigateTo('/');
            }
            decisions.resetForm.call(this);
        },
        
        openHistoryModal() {
            if (!this._isHandlingHistoryNavigation) {
                navigateTo('/history');
            }
            decisions.openHistoryModal.call(this);
        },
        
        loadConversation(id) {
            const conversation = this.conversationHistory.find(c => c.id === id);
            if (!conversation) return;
            
            // Check if this was triggered by our browser history handler
            if (!this._isHandlingHistoryNavigation) {
                // Only update history if this wasn't triggered by a popstate event
                navigateTo('/conversation/' + id, { conversationId: id });
            }
            
            // Always load the conversation content
            decisions.loadConversation.call(this, id);
        },
        
        startNewDecision() {
            if (!this._isHandlingHistoryNavigation) {
                navigateTo('/');
            }
            decisions.startNewDecision.call(this);
        },
        
        // Override showToastMessage to call the decision helper's method
        showToastMessage(message, showUndo = false) {
            decisions.showToastMessage.call(this, message, showUndo);
        },
        
        // Add undoLastAction to call the actions helper's method
        undoLastAction() {
            actions.undoLastAction.call(this);
        },
        
        // Make sure the decision helper can use the actions helper's methods
        saveTodos() {
            actions.saveTodos.call(this);
        }
    };
    
    return merged;
}

// Register the combined app with Alpine
Alpine.data('app', appData);

Alpine.start();
import 'htmx.org';
import '@fortawesome/fontawesome-free/css/all.min.css';
import 'animate.css';
import './tailwind.css';
import { CapacitorUpdater } from '@capgo/capacitor-updater'
CapacitorUpdater.notifyAppReady()
