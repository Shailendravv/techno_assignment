import React, { useState, useRef, useEffect } from 'react';
import { FaArrowDown } from 'react-icons/fa';
import {
  IoChevronDown,
  IoGlassesOutline,
  IoPulseOutline,
  IoPersonAddOutline,
  IoChevronUp
} from 'react-icons/io5';
import { MdOutlineContentCopy, MdOutlineAttachFile } from 'react-icons/md';

const ChatInterface = () => {
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState([
    {
      id: 1,
      sender: 'user',
      text: 'Testing',
      timestamp: 'Today'
    },
    {
      id: 2,
      sender: 'ai',
      text: 'Got it 👍 — looks like everything\'s working smoothly. Since you said "Testing," do you want me to show off a quick demo of something fun I can do, like generating a quiz, pulling fresh info from the web, or even creating an image?',
      timestamp: 'Today'
    },
    {
      id: 3,
      sender: 'user',
      text: 'sss',
      timestamp: 'Today'
    },
    {
      id: 4,
      sender: 'ai',
      text: 'All good, Shailendra — I see you\'re just tossing in a quick "sss" to keep the flow going. Want me to spin that into something fun, like a short poem or a quirky acronym? For example:',
      isCode: true,
      codeTitle: 'Code',
      codeContent: 'S - Spark ideas\nS - Shape them boldly',
      timestamp: 'Today'
    }
  ]);

  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      // In a real app, you would handle the file upload here
      console.log('Selected file:', file.name);
      // For demo purposes, we can add a system message or just clear the input
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = () => {
    if (message.trim() === '') return;

    const newMessage = {
      id: Date.now(),
      text: message,
      sender: 'user',
      timestamp: 'Today'
    };

    setMessages([...messages, newMessage]);
    setMessage('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-screen w-full max-w-5xl mx-auto relative font-sans text-[#333]">


      {/* Messages Area */}
      <div className="flex-grow overflow-y-auto px-4 pb-32 scrollbar-hide">
        <div className="max-w-3xl mx-auto">
          {/* Date Separator */}
          <div className="flex items-center my-8">
            <div className="flex-grow border-t border-gray-200"></div>
            <span className="px-4 text-xs font-medium text-gray-400">Today</span>
            <div className="flex-grow border-t border-gray-200"></div>
          </div>

          <div className="space-y-8">
            {messages.map((msg) => (
              <div key={msg.id} className={`flex flex-col ${msg.sender === 'user' ? 'items-end' : 'items-start'}`}>
                {msg.sender === 'user' ? (
                  <div className="bg-[#efedeb] rounded-[1.25rem] px-5 py-2.5 max-w-[80%] shadow-sm">
                    <p className="text-[15px] text-[#222]">{msg.text}</p>
                  </div>
                ) : (
                  <div className="w-full space-y-4">
                    <p className="text-[16px] leading-relaxed text-[#222]">
                      {msg.text}
                    </p>

                    {msg.isCode && (
                      <div className="relative group">
                        <div className="bg-[#1a1a1a] rounded-xl overflow-hidden shadow-xl">
                          <div className="flex items-center justify-between px-4 py-2 bg-[#2a2a2a] text-gray-300 text-xs">
                            <div className="flex items-center gap-2">
                              <span>{msg.codeTitle}</span>
                              <IoChevronUp className="text-gray-500" />
                            </div>
                            <button className="flex items-center gap-1.5 hover:text-white transition-colors">
                              <MdOutlineContentCopy />
                              Copy
                            </button>
                          </div>
                          <div className="p-5 font-mono text-[14px] text-gray-300 whitespace-pre bg-gradient-to-b from-[#1a1a1a] to-[#0a0a0a]">
                            {msg.codeContent}
                          </div>
                        </div>

                        {/* Scroll Down Button on Code Block */}
                        <div className="absolute left-1/2 -bottom-4 -translate-x-1/2">
                          <button className="w-8 h-8 rounded-full bg-white shadow-lg border border-gray-100 flex items-center justify-center text-gray-600 hover:bg-gray-50 transition-all">
                            <FaArrowDown className="text-xs" />
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Fixed Input Bar with Bottom Mask */}
      <div className="fixed bottom-0 left-0 right-0 pt-10 pb-6 px-4 bg-gradient-to-t from-[#f7f4f0] via-[#f7f4f0] to-transparent pointer-events-none z-20">
        <div className="max-w-3xl mx-auto pointer-events-auto">
          <div className="bg-white rounded-[2rem] p-2.5 shadow-[0_8px_30px_rgb(0,0,0,0.08)] border border-gray-100 flex items-center gap-2">
            <button
              onClick={handleUploadClick}
              className="w-10 h-10 rounded-full flex items-center justify-center text-gray-400 hover:bg-gray-50 transition-colors"
            >
              <MdOutlineAttachFile size={22} />
            </button>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              className="hidden"
              accept="image/*,application/pdf,.doc,.docx,.txt"
            />

            <textarea
              rows="1"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Message Copilot"
              className="flex-grow bg-transparent border-none outline-none text-[15px] text-[#222] placeholder-gray-400 py-2 px-2 resize-none max-h-32"
              style={{ minHeight: '24px' }}
            />
          </div>
        </div>
      </div>

    </div>
  );
};

export default ChatInterface;
