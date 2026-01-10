// Actions/Todo list functionality
export function actionsHelper() {
    return {
        todos: [],
        completedTodos: [],
        newTodo: '',
        draggedTodoId: null,
        touchStartY: null,
        touchStartX: null,
        touchElement: null,
        longPressTimer: null,
        isDragging: false,
        currentTouchTarget: null,
        dragClone: null,
        touchOffsetX: 0,
        touchOffsetY: 0,
        originalPosition: null,
        placeholderElement: null,
        lastDeletedTodo: null,
        lastDeletedTodoWasCompleted: false,
        _toastTimeout: null, // For tracking the toast timeout

        init() {
            this.loadTodos();
        },

        loadTodos() {
            const savedTodos = localStorage.getItem('todos');
            const savedCompletedTodos = localStorage.getItem('completedTodos');
            
            if (savedTodos) {
                this.todos = JSON.parse(savedTodos);
            }
            
            if (savedCompletedTodos) {
                this.completedTodos = JSON.parse(savedCompletedTodos);
            }
        },
        
        saveTodos() {
            localStorage.setItem('todos', JSON.stringify(this.todos));
            localStorage.setItem('completedTodos', JSON.stringify(this.completedTodos));
        },
        
        addTodo() {
            if (this.newTodo.trim() === '') return;
            
            this.todos.unshift({
                id: this.generateId(),
                text: this.newTodo.trim(),
                createdAt: new Date().toISOString(),
                hasDiscussion: false,
                discussionId: null
            });
            
            this.newTodo = '';
            this.saveTodos();
            this.showToastMessage('Todo added');
        },
        
        // Drag and drop methods
        startDrag(event, todoId) {
            this.draggedTodoId = todoId;
            event.dataTransfer.effectAllowed = 'move';
            
            // Find the parent li element
            const listItem = event.target.closest('li');
            
            // Add a class to the list item for styling
            listItem.classList.add('dragging');
            
            // Set the drag image to be the list item
            event.dataTransfer.setDragImage(listItem, 20, 20);
        },
        
        endDrag(event) {
            this.draggedTodoId = null;
            
            // Find the parent li element
            const listItem = event.target.closest('li');
            
            // Remove the dragging class
            if (listItem) {
                listItem.classList.remove('dragging');
            }
        },
        
        onDragOver(event) {
            event.preventDefault();
            
            // Simplified - directly add the class without conditional check
            event.currentTarget.classList.add('drag-over');
            
            return false;
        },
        
        onDragLeave(event) {
            // Remove the drag-over class when leaving the element
            event.currentTarget.classList.remove('drag-over');
        },
        
        onDrop(event, targetTodoId) {
            event.preventDefault();
            
            // Remove the drag-over class
            event.currentTarget.classList.remove('drag-over');
            
            // Don't do anything if we're dropping onto the same item
            if (this.draggedTodoId === targetTodoId) {
                return;
            }
            
            // Find the indices of the dragged and target todos
            const draggedIndex = this.todos.findIndex(todo => todo.id === this.draggedTodoId);
            const targetIndex = this.todos.findIndex(todo => todo.id === targetTodoId);
            
            if (draggedIndex !== -1 && targetIndex !== -1) {
                // Remove the dragged item
                const draggedTodo = this.todos.splice(draggedIndex, 1)[0];
                
                // Insert it at the target position
                this.todos.splice(targetIndex, 0, draggedTodo);
                
                // Save the updated order
                this.saveTodos();
                this.showToastMessage('Todo order updated');
            }
        },
        
        completeTodo(todoId) {
            const todoIndex = this.todos.findIndex(todo => todo.id === todoId);
            
            if (todoIndex !== -1) {
                const completedTodo = this.todos.splice(todoIndex, 1)[0];
                
                // Clear any pending auto-save timer
                if (completedTodo._autoSaveTimer) {
                    clearTimeout(completedTodo._autoSaveTimer);
                    delete completedTodo._autoSaveTimer;
                }
                
                completedTodo.completedAt = new Date().toISOString();
                this.completedTodos.unshift(completedTodo);
                this.saveTodos();
                this.showToastMessage('Todo completed');
            }
        },
        
        deleteTodo(todoId, isCompleted = false) {
            let deletedTodo = null;
            
            if (isCompleted) {
                // Find the todo before removing it to clear any timers
                const todoIndex = this.completedTodos.findIndex(todo => todo.id === todoId);
                if (todoIndex !== -1) {
                    deletedTodo = { ...this.completedTodos[todoIndex] };
                    const todo = this.completedTodos[todoIndex];
                    if (todo._autoSaveTimer) {
                        clearTimeout(todo._autoSaveTimer);
                    }
                }
                
                this.completedTodos = this.completedTodos.filter(todo => todo.id !== todoId);
            } else {
                // Find the todo before removing it to clear any timers
                const todoIndex = this.todos.findIndex(todo => todo.id === todoId);
                if (todoIndex !== -1) {
                    deletedTodo = { ...this.todos[todoIndex] };
                    const todo = this.todos[todoIndex];
                    if (todo._autoSaveTimer) {
                        clearTimeout(todo._autoSaveTimer);
                    }
                }
                
                this.todos = this.todos.filter(todo => todo.id !== todoId);
            }
            
            // Save the deleted todo for potential undo
            if (deletedTodo) {
                this.lastDeletedTodo = deletedTodo;
                this.lastDeletedTodoWasCompleted = isCompleted;
            }
            
            this.saveTodos();
            this.showToastMessage('Todo deleted', true);
        },
        
        undoLastAction() {
            if (this.lastDeletedTodo) {
                // Restore the deleted todo
                if (this.lastDeletedTodoWasCompleted) {
                    this.completedTodos.push(this.lastDeletedTodo);
                } else {
                    this.todos.push(this.lastDeletedTodo);
                }
                
                // Clear the last deleted todo
                this.lastDeletedTodo = null;
                this.lastDeletedTodoWasCompleted = false;
                
                this.saveTodos();
                this.showToastMessage('Todo restored', false);
            }
        },
        
        restoreTodo(todoId) {
            const todoIndex = this.completedTodos.findIndex(todo => todo.id === todoId);
            
            if (todoIndex !== -1) {
                const restoredTodo = this.completedTodos.splice(todoIndex, 1)[0];
                
                // Clear any pending auto-save timer
                if (restoredTodo._autoSaveTimer) {
                    clearTimeout(restoredTodo._autoSaveTimer);
                    delete restoredTodo._autoSaveTimer;
                }
                
                delete restoredTodo.completedAt;
                this.todos.push(restoredTodo);
                this.saveTodos();
                this.showToastMessage('Todo restored');
            }
        },

        // Helper methods that need to be shared with the main app
        generateId() {
            return Date.now().toString(36) + Math.random().toString(36).substr(2, 5);
        },
        
        showToastMessage(message, showUndo = false) {
            if (this.$parent && typeof this.$parent.showToastMessage === 'function') {
                this.$parent.showToastMessage(message, showUndo);
            } else {
                // Clear any existing timeout to prevent premature hiding
                if (this._toastTimeout) {
                    clearTimeout(this._toastTimeout);
                    this._toastTimeout = null;
                }
                
                this.toastMessage = message;
                this.showUndoButton = showUndo;
                this.showToast = true;
                
                this._toastTimeout = setTimeout(() => {
                    this.showToast = false;
                    this._toastTimeout = null;
                }, 5000);
            }
        },

        startInlineEdit(todo) {
            // Store the original text in case we need to cancel
            todo._originalText = todo.text;
            
            // Clear any existing auto-save timer for this todo
            if (todo._autoSaveTimer) {
                clearTimeout(todo._autoSaveTimer);
                todo._autoSaveTimer = null;
            }
        },
        
        updateTodoText(event, todo) {
            // Update the todo text in real-time as the user types
            todo.text = event.target.textContent;
            
            // Clear any existing auto-save timer
            if (todo._autoSaveTimer) {
                clearTimeout(todo._autoSaveTimer);
            }
            
            // Set a new timer to auto-save after 2 seconds of inactivity
            todo._autoSaveTimer = setTimeout(() => {
                this.autoSaveTodo(event, todo);
            }, 2000);
        },
        
        autoSaveTodo(event, todo) {
            const newText = event.target.textContent.trim();
            
            // Don't save empty todos
            if (newText === '') {
                return;
            }
            
            // Only save if the text has actually changed
            if (newText !== todo._originalText) {
                todo.text = newText;
                delete todo._originalText;
                delete todo._autoSaveTimer;
                this.saveTodos();
                
                // Show a subtle indicator that the todo was saved
                this.showToastMessage('Todo saved');
            }
        },
        
        saveTodoText(event, todo) {
            // Clear any pending auto-save timer
            if (todo._autoSaveTimer) {
                clearTimeout(todo._autoSaveTimer);
                todo._autoSaveTimer = null;
            }
            
            const newText = event.target.textContent.trim();
            
            // Don't save empty todos
            if (newText === '') {
                event.target.textContent = todo._originalText || '';
                return;
            }
            
            todo.text = newText;
            delete todo._originalText;
            this.saveTodos();
            
            // Remove focus from the element
            event.target.blur();
        },
        
        cancelTodoEdit(event, todo) {
            // Clear any pending auto-save timer
            if (todo._autoSaveTimer) {
                clearTimeout(todo._autoSaveTimer);
                todo._autoSaveTimer = null;
            }
            
            // Restore the original text
            event.target.textContent = todo._originalText || todo.text;
            delete todo._originalText;
            
            // Remove focus from the element
            event.target.blur();
        },

        // Mobile touch handlers
        handleTouchStart(event, todoId) {
            // Store the initial touch position
            this.touchStartY = event.touches[0].clientY;
            this.touchStartX = event.touches[0].clientX;
            
            // Find the grip element and the parent li element
            this.touchElement = event.currentTarget.closest('li');
            this.currentTouchTarget = todoId;
            
            // Calculate the offset from the touch point to the element's top-left corner
            const rect = this.touchElement.getBoundingClientRect();
            this.touchOffsetX = this.touchStartX - rect.left;
            this.touchOffsetY = this.touchStartY - rect.top;
            
            // Store the original position for creating a placeholder later
            this.originalPosition = {
                width: rect.width,
                height: rect.height,
                left: rect.left,
                top: rect.top
            };
            
            // Set a timer for long press (500ms)
            this.longPressTimer = setTimeout(() => {
                this.isDragging = true;
                this.draggedTodoId = todoId;
                
                // Create a clone of the element to move with the finger
                this.createDragClone();
                
                // Create a placeholder in the original position
                this.touchElement.classList.add('drag-placeholder');
                this.placeholderElement = this.touchElement;
                
                // Vibrate if supported
                if (navigator.vibrate) {
                    navigator.vibrate(50);
                }
                
                // Show toast to indicate drag mode
                this.showToastMessage('Drag to reorder');
            }, 300); // Reduced from 500ms to 300ms for faster response
        },
        
        createDragClone() {
            // Create a clone of the element
            this.dragClone = this.touchElement.cloneNode(true);
            
            // Style the clone
            this.dragClone.classList.add('touch-dragging');
            this.dragClone.classList.remove('drag-placeholder');
            
            // Position the clone at the original position
            this.dragClone.style.position = 'fixed';
            this.dragClone.style.left = `${this.originalPosition.left}px`;
            this.dragClone.style.top = `${this.originalPosition.top}px`;
            this.dragClone.style.width = `${this.originalPosition.width}px`;
            this.dragClone.style.margin = '0';
            this.dragClone.style.zIndex = '9999';
            
            // Add the clone to the document body
            document.body.appendChild(this.dragClone);
            
            // Move the clone to the current touch position
            this.updateDragClonePosition(this.touchStartX, this.touchStartY);
        },
        
        updateDragClonePosition(touchX, touchY) {
            if (!this.dragClone) return;
            
            // Calculate the new position based on the touch position and the initial offset
            const left = touchX - this.touchOffsetX;
            const top = touchY - this.touchOffsetY;
            
            // Update the clone's position
            this.dragClone.style.left = `${left}px`;
            this.dragClone.style.top = `${top}px`;
        },
        
        handleTouchMove(event) {
            if (!this.isDragging) {
                // If not in dragging mode yet, check if we should cancel the long press
                const touchY = event.touches[0].clientY;
                const touchX = event.touches[0].clientX;
                
                // If moved more than 10px in any direction before long press activated, cancel it
                if (Math.abs(touchY - this.touchStartY) > 10 || Math.abs(touchX - this.touchStartX) > 10) {
                    clearTimeout(this.longPressTimer);
                }
                return;
            }
            
            // Prevent default only when we're actually dragging
            event.preventDefault();
            event.stopPropagation();
            
            // Get the current touch position
            const touchX = event.touches[0].clientX;
            const touchY = event.touches[0].clientY;
            
            // Update the position of the drag clone
            this.updateDragClonePosition(touchX, touchY);
            
            // Get the element under the touch point (excluding the clone)
            const elementsUnderTouch = document.elementsFromPoint(touchX, touchY);
            const targetLi = elementsUnderTouch.find(el => 
                el.tagName === 'LI' && 
                !el.classList.contains('touch-dragging') && 
                !el.classList.contains('drag-placeholder')
            );
            
            if (targetLi) {
                // Get the todo ID from the element
                const targetTodoId = targetLi.getAttribute('data-id');
                
                if (targetTodoId && targetTodoId !== this.draggedTodoId) {
                    // Add visual feedback
                    document.querySelectorAll('li').forEach(li => {
                        if (li !== this.placeholderElement) {
                            li.classList.remove('drag-over');
                        }
                    });
                    targetLi.classList.add('drag-over');
                    
                    // Update the current touch target
                    this.currentTouchTarget = targetTodoId;
                }
            }
        },
        
        handleTouchEnd(event, todoId) {
            // Clear the long press timer
            clearTimeout(this.longPressTimer);
            
            // If we were dragging
            if (this.isDragging) {
                // Remove the drag clone
                if (this.dragClone) {
                    this.dragClone.remove();
                    this.dragClone = null;
                }
                
                // Remove the placeholder class
                if (this.placeholderElement) {
                    this.placeholderElement.classList.remove('drag-placeholder');
                    this.placeholderElement = null;
                }
                
                // Remove visual feedback from all items
                document.querySelectorAll('li').forEach(li => {
                    li.classList.remove('drag-over');
                });
                
                // If we have a valid drop target
                if (this.currentTouchTarget && this.currentTouchTarget !== this.draggedTodoId) {
                    this.onDrop(event, this.currentTouchTarget);
                }
                
                // Reset state
                this.isDragging = false;
            }
            
            this.draggedTodoId = null;
            this.touchStartY = null;
            this.touchStartX = null;
            this.touchElement = null;
            this.currentTouchTarget = null;
            this.touchOffsetX = 0;
            this.touchOffsetY = 0;
            this.originalPosition = null;
        }
    };
} 