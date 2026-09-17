import React, { useState, useRef, useEffect } from 'react';
import { IoChevronUp } from 'react-icons/io5';
import { MdOutlineContentCopy } from 'react-icons/md';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import Header from './Header';
import InputBar from './InputBar';
import { useMutation } from '@tanstack/react-query';
import { askQuestion } from '../services/api';

const CONFIDENCE_STYLES = {
  high: 'bg-green-50 text-green-700 border-green-100',
  medium: 'bg-amber-50 text-amber-700 border-amber-100',
  low: 'bg-orange-50 text-orange-700 border-orange-100',
  no_match: 'bg-gray-100 text-gray-500 border-gray-200',
};

// Maps markdown elements to the existing chat bubble's typography, so the
// AI's numbered steps, code spans, and emphasis render instead of showing
// raw "1." / "`...`" / "**...**" characters.
const MARKDOWN_COMPONENTS = {
  p: ({ children }) => <p className="mb-3 last:mb-0">{children}</p>,
  ol: ({ children }) => <ol className="list-decimal pl-5 space-y-1.5 mb-3 last:mb-0">{children}</ol>,
  ul: ({ children }) => <ul className="list-disc pl-5 space-y-1.5 mb-3 last:mb-0">{children}</ul>,
  li: ({ children }) => <li className="pl-1">{children}</li>,
  strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
  code: ({ children }) => (
    <code className="bg-gray-100 text-[#c7254e] rounded px-1.5 py-0.5 text-[0.9em] font-mono">
      {children}
    </code>
  ),
  a: ({ children, href }) => (
    <a href={href} target="_blank" rel="noreferrer" className="text-blue-600 underline">
      {children}
    </a>
  ),
};

const ChatInterface = () => {
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState([]);

  const messagesEndRef = useRef(null);

  // Mutation for asking the runbook agent a question
  const mutation = useMutation({
    mutationFn: (question) => askQuestion(question),
    onSuccess: (data) => {
      const aiResponse = {
        id: Date.now() + 1,
        sender: 'ai',
        text: data.answer,
        confidence: data.confidence,
        citedDocIds: data.cited_doc_ids || [],
        timestamp: 'Today'
      };
      setMessages((prev) => [...prev, aiResponse]);
    },
    onError: (error) => {
      const errorMessage = {
        id: Date.now() + 1,
        sender: 'ai',
        text: `Error: ${error.message}`,
        timestamp: 'Today'
      };
      setMessages((prev) => [...prev, errorMessage]);
    }
  });

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
      timestamp: 'Today',
    };

    setMessages([...messages, newMessage]);
    mutation.mutate(message);
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
      <Header />


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
                  <div className="w-full space-y-3">
                    <div className="text-[16px] leading-relaxed text-[#222]">
                      <ReactMarkdown remarkPlugins={[remarkGfm]} components={MARKDOWN_COMPONENTS}>
                        {msg.text}
                      </ReactMarkdown>
                    </div>

                    {msg.confidence && (
                      <div className="flex flex-wrap items-center gap-2">
                        <span
                          className={`text-xs font-medium px-2.5 py-1 rounded-full border ${
                            CONFIDENCE_STYLES[msg.confidence] || CONFIDENCE_STYLES.no_match
                          }`}
                        >
                          confidence: {msg.confidence}
                        </span>
                        {msg.citedDocIds?.map((docId) => (
                          <span
                            key={docId}
                            className="text-xs font-medium px-2.5 py-1 rounded-full border bg-gray-50 text-gray-600 border-gray-200"
                          >
                            {docId}
                          </span>
                        ))}
                      </div>
                    )}

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
      <InputBar
        message={message}
        setMessage={setMessage}
        handleSend={handleSend}
        handleKeyDown={handleKeyDown}
        isLoading={mutation.isPending}
      />

    </div>
  );
};

export default ChatInterface;
