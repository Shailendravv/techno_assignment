import React from 'react';
import { IoPersonAddOutline, IoChevronDown } from 'react-icons/io5';

const Header = () => {
  return (
    <header className="sticky top-0 z-30 w-full bg-[#f7f4f0]/80 backdrop-blur-md border-b border-gray-200/50 px-4 py-3">
      <div className="max-w-3xl mx-auto flex items-center justify-between w-full">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-700 flex items-center justify-center text-white shadow-lg shadow-blue-500/20">
            {/* This is a placeholder for the logo */}
            <svg viewBox="0 0 24 24" className="w-5 h-5 fill-current">
              <path d="M12 2L4.5 20.29l.71.71L12 18l6.79 3 .71-.71z" />
            </svg>
          </div>
          <div>
            <h1 className="font-bold text-[17px] text-gray-900 leading-tight">AI Assistant</h1>
            <p className="text-[11px] text-gray-500 font-medium">Powered by Copilot</p>
          </div>
        </div>

      </div>
    </header>
  );
};

export default Header;
