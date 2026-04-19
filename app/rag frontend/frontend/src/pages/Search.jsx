import { useState } from "react";
import { Search as SearchIcon, Filter, FileText, ChevronRight, Loader2 } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

export default function Search() {
  const [query, setQuery] = useState("");
  const [searchType, setSearchType] = useState("content");
  const [isSearching, setIsSearching] = useState(false);
  const [results, setResults] = useState(null);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    
    setIsSearching(true);
    setResults(null);

    try {
      if (searchType === "title") {
        // If searching by title, the backend vector search only supports content.
        // We will fetch all documents and filter by filename on the frontend.
        const res = await fetch("/pdfs");
        if (res.ok) {
          const allDocs = await res.json();
          const filtered = allDocs.filter(doc => doc.toLowerCase().includes(query.toLowerCase()));
          
          setResults(filtered.map((doc, i) => ({
            id: `title_${i}`,
            title: doc.split('/').pop(),
            document: doc,
            page: "1",
            chunk: "N/A",
            snippet: "Matched by filename",
            relevance: 1.0
          })));
        } else {
          setResults([]);
        }
      } else {
        // Semantic search using the backend API
        // Do NOT pass search_type to backend since it may cause Pydantic to ignore the query or throw 422
        const res = await fetch("/search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: query, top_k: 5 }), 
        });
        
        if (res.ok) {
          const data = await res.json();
          
          // Handle both {"results": [...]} and [...] response formats safely
          let resultsArray = [];
          if (Array.isArray(data)) {
            resultsArray = data;
          } else if (data && Array.isArray(data.results)) {
            resultsArray = data.results;
          }
          
          const mappedResults = resultsArray.map((r, i) => ({
            id: r.id || i,
            title: r.file_name ? r.file_name.split('/').pop() : `Result ${i+1}`,
            document: r.file_name || "Unknown",
            page: r.page_number || "N/A",
            chunk: r.chunk_id || "N/A",
            snippet: r.content || "No content available",
            relevance: r.rrf_score || r.score || 0.0
          }));
          setResults(mappedResults);
        } else {
          setResults([]);
        }
      }
    } catch (err) {
      console.error("Search API failed", err);
      setResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <div className="page-content" style={{ maxWidth: '900px', margin: '0 auto', display: 'flex', flexDirection: 'column', height: '100%' }}>
      
      <div style={{ textAlign: 'center', margin: '3rem 0 2rem 0' }}>
        <h1 style={{ fontSize: '2.5rem', fontWeight: 700, marginBottom: '1rem', background: 'linear-gradient(to right, var(--foreground), var(--muted-foreground))', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
          Semantic Search
        </h1>
        <p style={{ color: 'var(--muted-foreground)', fontSize: '1.1rem', maxWidth: '600px', margin: '0 auto' }}>
          Instantly find relevant information across all your uploaded documents using advanced vector search.
        </p>
      </div>

      <form onSubmit={handleSearch} style={{ position: 'relative', marginBottom: '2rem', zIndex: 10 }}>
        <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
          <SearchIcon style={{ position: 'absolute', left: '1.25rem', color: 'var(--muted-foreground)' }} size={24} />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question or search for keywords..."
            style={{
              width: '100%',
              padding: '1.25rem 4rem 1.25rem 3.5rem',
              fontSize: '1.1rem',
              backgroundColor: 'var(--card)',
              border: '1px solid var(--border)',
              borderRadius: '2rem',
              color: 'var(--foreground)',
              boxShadow: 'var(--shadow-lg)',
              outline: 'none',
              transition: 'all 0.2s ease',
            }}
            onFocus={(e) => {
              e.target.style.borderColor = 'var(--primary)';
              e.target.style.boxShadow = '0 0 0 2px rgba(59, 130, 246, 0.2)';
            }}
            onBlur={(e) => {
              e.target.style.borderColor = 'var(--border)';
              e.target.style.boxShadow = 'var(--shadow-lg)';
            }}
          />
          <button 
            type="submit"
            className="btn btn-primary"
            style={{ position: 'absolute', right: '0.5rem', borderRadius: '1.5rem', padding: '0.75rem 1.5rem' }}
            disabled={isSearching || !query.trim()}
          >
            {isSearching ? <Loader2 size={20} className="animate-spin" /> : "Search"}
          </button>
        </div>
        
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '1rem' }}>
          <div style={{ display: 'flex', background: 'var(--card)', padding: '4px', borderRadius: '8px', border: '1px solid var(--border)' }}>
            <button
              type="button"
              onClick={() => setSearchType("content")}
              style={{
                padding: '6px 12px',
                borderRadius: '6px',
                border: 'none',
                background: searchType === "content" ? 'var(--primary)' : 'transparent',
                color: searchType === "content" ? 'var(--primary-foreground)' : 'var(--muted-foreground)',
                fontSize: '0.875rem',
                fontWeight: 500,
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
            >
              By Content
            </button>
            <button
              type="button"
              onClick={() => setSearchType("title")}
              style={{
                padding: '6px 12px',
                borderRadius: '6px',
                border: 'none',
                background: searchType === "title" ? 'var(--primary)' : 'transparent',
                color: searchType === "title" ? 'var(--primary-foreground)' : 'var(--muted-foreground)',
                fontSize: '0.875rem',
                fontWeight: 500,
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
            >
              By Title
            </button>
          </div>

          <button type="button" className="btn btn-ghost" style={{ fontSize: '0.875rem' }}>
            <Filter size={16} /> Filters
          </button>
        </div>
      </form>

      <div style={{ flex: 1 }}>
        <AnimatePresence mode="wait">
          {isSearching ? (
             <motion.div 
               key="searching"
               initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
               style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '4rem 0', color: 'var(--muted-foreground)' }}
             >
               <Loader2 size={40} className="animate-spin" style={{ color: 'var(--primary)', marginBottom: '1rem' }} />
               <p>Searching through documents...</p>
             </motion.div>
          ) : results ? (
            <motion.div 
              key="results"
              initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
              style={{ display: 'flex', flexDirection: 'column', gap: '1rem', paddingBottom: '2rem' }}
            >
              <h2 style={{ fontSize: '1.25rem', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                Found {results.length} results
              </h2>
              
              {results.map((result, i) => (
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.1 }}
                  key={result.id} 
                  className="card"
                  style={{ cursor: 'pointer', transition: 'transform 0.2s, box-shadow 0.2s' }}
                  onMouseOver={(e) => {
                    e.currentTarget.style.transform = 'translateY(-2px)';
                    e.currentTarget.style.boxShadow = 'var(--shadow-lg)';
                    e.currentTarget.style.borderColor = 'rgba(59, 130, 246, 0.3)';
                  }}
                  onMouseOut={(e) => {
                    e.currentTarget.style.transform = 'none';
                    e.currentTarget.style.boxShadow = 'var(--shadow)';
                    e.currentTarget.style.borderColor = 'var(--border)';
                  }}
                >
                  <div className="card-content" style={{ display: 'flex', gap: '1.5rem', alignItems: 'flex-start' }}>
                    <div style={{ background: 'rgba(59, 130, 246, 0.1)', padding: '0.75rem', borderRadius: '12px', flexShrink: 0 }}>
                      <FileText size={24} color="var(--primary)" />
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.5rem' }}>
                        <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--primary)', wordBreak: 'break-all' }}>{result.title}</h3>
                        <span style={{ fontSize: '0.75rem', background: 'var(--muted)', padding: '4px 10px', borderRadius: '12px', color: 'var(--accent)', fontWeight: 600 }}>
                          Score: {result.relevance > 0 ? result.relevance.toFixed(3) : "N/A"}
                        </span>
                      </div>
                      <p style={{ fontSize: '0.875rem', color: 'var(--muted-foreground)', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                        <span style={{ background: 'var(--card)', border: '1px solid var(--border)', padding: '2px 8px', borderRadius: '4px' }}>Doc: {result.document}</span>
                        {result.page !== "N/A" && (
                          <span style={{ background: 'var(--card)', border: '1px solid var(--border)', padding: '2px 8px', borderRadius: '4px' }}>Page: {result.page}</span>
                        )}
                        {result.chunk !== "N/A" && (
                          <span style={{ background: 'var(--card)', border: '1px solid var(--border)', padding: '2px 8px', borderRadius: '4px' }}>Chunk: {result.chunk}</span>
                        )}
                      </p>
                      <p style={{ color: 'var(--foreground)', lineHeight: 1.6, fontSize: '0.95rem' }}>
                        {result.snippet}
                      </p>
                    </div>
                    <ChevronRight size={20} color="var(--muted-foreground)" style={{ alignSelf: 'center', flexShrink: 0 }} />
                  </div>
                </motion.div>
              ))}
            </motion.div>
          ) : (
             <motion.div 
               key="empty"
               initial={{ opacity: 0 }} animate={{ opacity: 1 }}
               style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '4rem 0', color: 'var(--muted-foreground)', opacity: 0.5 }}
             >
               <SearchIcon size={48} style={{ marginBottom: '1rem', opacity: 0.5 }} />
               <p>Enter a query to search across all processed documents</p>
             </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
