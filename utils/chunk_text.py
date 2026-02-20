def chunk_text(text, max_length=512):
    """
    Split long text into manageable chunks.
    Simple implementation splitting by whitespace to avoid breaking words,
    but respecting max token length roughly (assuming avg token length).
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    # Rough estimation: 1 word ~ 1.3 tokens? Or simply chars?
    # User said max_length=512. T5 constraint.
    # We'll be conservative and say 400 words per chunk to stay under 512 tokens usually.
    # Or just use character count ~ 2000 chars.
    
    max_chars = 2000
    
    for word in words:
        if current_length + len(word) + 1 > max_chars:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks
