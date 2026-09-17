import React from 'react';
import { IoSend, IoReload } from 'react-icons/io5';

const InputBar = ({ message, setMessage, handleSend, handleKeyDown, isLoading }) => {
    return (
        <div className="fixed bottom-0 left-0 right-0 pt-10 pb-6 px-4 bg-gradient-to-t from-[#f7f4f0] via-[#f7f4f0] to-transparent pointer-events-none z-20">
            <div className="max-w-3xl mx-auto pointer-events-auto">
                <div className="bg-white rounded-[2rem] p-2.5 shadow-[0_8px_30px_rgb(0,0,0,0.08)] border border-gray-100 flex items-center gap-2">
                    <div className="flex-grow flex flex-col gap-1">
                        <textarea
                            rows="1"
                            value={message}
                            onChange={(e) => setMessage(e.target.value)}
                            onKeyDown={handleKeyDown}
                            placeholder="Ask about a runbook..."
                            className="w-full bg-transparent border-none outline-none text-[15px] text-[#222] placeholder-gray-400 py-2 px-1 resize-none max-h-32"
                            style={{ minHeight: '24px' }}
                            disabled={isLoading}
                        />
                    </div>

                    <button
                        onClick={handleSend}
                        disabled={isLoading || message.trim() === ''}
                        className={`w-10 h-10 rounded-full flex items-center justify-center transition-all ${
                            isLoading || message.trim() === ''
                                ? 'bg-gray-100 text-gray-300 cursor-not-allowed'
                                : 'bg-[#1a1a1a] text-white hover:bg-black active:scale-95 shadow-md'
                        }`}
                    >
                        {isLoading ? (
                            <IoReload className="animate-spin" size={20} />
                        ) : (
                            <IoSend size={18} className="ml-0.5" />
                        )}
                    </button>
                </div>
            </div>
        </div>
    );
};

export default InputBar;
