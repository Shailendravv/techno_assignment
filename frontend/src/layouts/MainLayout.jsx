
const MainLayout = ({ children }) => {
  return (
    <div className="min-h-screen selection:bg-cyan-500 selection:text-white">
      {children}
    </div>
  );
};

export default MainLayout;
