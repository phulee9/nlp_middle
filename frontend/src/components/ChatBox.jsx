import { useState } from 'react';

export default function ChatBox({ disabled, onAsk }) {
    const [question, setQuestion] = useState('');

    function handleSubmit(e) {
        e.preventDefault();
        if (!question.trim() || disabled) return;
        onAsk(question.trim());
        setQuestion('');
    }

    function handleKeyDown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
        }
    }

    return (
        <form className="chat-form" onSubmit={handleSubmit}>
            <div className="chat-input-wrap">
                <input
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder={disabled ? 'Upload a PDF to start chatting…' : 'Ask anything about your document…'}
                    disabled={disabled}
                    autoComplete="off"
                />
                <button
                    type="submit"
                    className="chat-send-btn"
                    disabled={disabled || !question.trim()}
                    aria-label="Send"
                >
                    <svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">
                        <path d="M14.5 8L2 14l2.5-6H14.5zM2 2l12.5 6H4.5L2 2z" />
                    </svg>
                </button>
            </div>
        </form>
    );
}
