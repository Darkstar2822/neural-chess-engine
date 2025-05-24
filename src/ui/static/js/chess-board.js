// Chess Board JavaScript with Drag & Drop
class ChessBoard {
    constructor() {
        this.selectedSquare = null;
        this.gameState = {
            fen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
            humanColor: 'white',
            gameActive: false,
            currentTurn: 'white'
        };
        this.draggedPiece = null;
        this.draggedFrom = null;
        
        this.initializeBoard();
        this.setupEventListeners();
    }

    initializeBoard() {
        const boardElement = document.getElementById('chess-board');
        boardElement.innerHTML = '';
        
        for (let rank = 8; rank >= 1; rank--) {
            for (let file = 1; file <= 8; file++) {
                const square = document.createElement('div');
                const squareId = String.fromCharCode(96 + file) + rank;
                square.id = squareId;
                square.className = `square ${(rank + file) % 2 === 0 ? 'dark' : 'light'}`;
                
                // Add coordinates
                if (file === 8) {
                    const rankCoord = document.createElement('div');
                    rankCoord.className = 'coords rank-coord';
                    rankCoord.textContent = rank;
                    square.appendChild(rankCoord);
                }
                if (rank === 1) {
                    const fileCoord = document.createElement('div');
                    fileCoord.className = 'coords file-coord';
                    fileCoord.textContent = String.fromCharCode(96 + file);
                    square.appendChild(fileCoord);
                }
                
                square.addEventListener('click', (e) => this.handleSquareClick(e));
                square.addEventListener('dragover', (e) => this.handleDragOver(e));
                square.addEventListener('drop', (e) => this.handleDrop(e));
                
                boardElement.appendChild(square);
            }
        }
        
        this.updateBoardFromFEN(this.gameState.fen);
    }

    setupEventListeners() {
        document.addEventListener('dragstart', (e) => this.handleDragStart(e));
        document.addEventListener('dragend', (e) => this.handleDragEnd(e));
    }

    updateBoardFromFEN(fen) {
        const [position] = fen.split(' ');
        const ranks = position.split('/');
        
        // Clear all pieces
        document.querySelectorAll('.piece').forEach(piece => piece.remove());
        
        ranks.forEach((rank, rankIndex) => {
            let fileIndex = 0;
            for (let char of rank) {
                if (isNaN(char)) {
                    const squareId = String.fromCharCode(97 + fileIndex) + (8 - rankIndex);
                    const square = document.getElementById(squareId);
                    if (square) {
                        const piece = this.createPieceElement(char);
                        if (piece) {
                            square.appendChild(piece);
                        }
                    }
                    fileIndex++;
                } else {
                    fileIndex += parseInt(char);
                }
            }
        });
    }

    createPieceElement(pieceChar) {
        const pieceImageMap = {
            'K': '/static/images/wk.png',  // White King
            'Q': '/static/images/wq.png',  // White Queen
            'R': '/static/images/wr.png',  // White Rook
            'B': '/static/images/wb.png',  // White Bishop
            'N': '/static/images/wn.png',  // White Knight
            'P': '/static/images/wp.png',  // White Pawn
            'k': '/static/images/bk.png',  // Black King
            'q': '/static/images/bq.png',  // Black Queen
            'r': '/static/images/br.png',  // Black Rook
            'b': '/static/images/bb.png',  // Black Bishop
            'n': '/static/images/bn.png',  // Black Knight
            'p': '/static/images/bp.png'   // Black Pawn
        };
        
        if (!pieceImageMap[pieceChar]) return null;
        
        const piece = document.createElement('div');
        piece.className = 'piece';
        piece.style.backgroundImage = `url('${pieceImageMap[pieceChar]}')`;
        piece.draggable = true;
        piece.dataset.piece = pieceChar;
        piece.title = this.getPieceName(pieceChar); // Tooltip for accessibility
        
        return piece;
    }

    getPieceName(pieceChar) {
        const pieceNames = {
            'K': 'White King', 'Q': 'White Queen', 'R': 'White Rook', 
            'B': 'White Bishop', 'N': 'White Knight', 'P': 'White Pawn',
            'k': 'Black King', 'q': 'Black Queen', 'r': 'Black Rook', 
            'b': 'Black Bishop', 'n': 'Black Knight', 'p': 'Black Pawn'
        };
        return pieceNames[pieceChar] || 'Chess Piece';
    }

    handleSquareClick(e) {
        e.preventDefault();
        const square = e.currentTarget;
        const squareId = square.id;
        
        if (this.selectedSquare) {
            if (this.selectedSquare === squareId) {
                this.clearSelection();
            } else {
                this.attemptMove(this.selectedSquare, squareId);
            }
        } else {
            const piece = square.querySelector('.piece');
            if (piece && this.canMovePiece(piece)) {
                this.selectSquare(squareId);
            }
        }
    }

    handleDragStart(e) {
        if (!e.target.classList.contains('piece')) return;
        
        const piece = e.target;
        const square = piece.parentElement;
        
        if (!this.canMovePiece(piece)) {
            e.preventDefault();
            return;
        }
        
        this.draggedPiece = piece;
        this.draggedFrom = square.id;
        piece.classList.add('dragging');
        
        // Show legal moves
        this.showLegalMoves(square.id);
        
        e.dataTransfer.effectAllowed = 'move';
        e.dataTransfer.setData('text/plain', '');
    }

    handleDragOver(e) {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
    }

    handleDrop(e) {
        e.preventDefault();
        
        if (!this.draggedPiece || !this.draggedFrom) return;
        
        const targetSquare = e.currentTarget;
        const targetSquareId = targetSquare.id;
        
        if (this.draggedFrom !== targetSquareId) {
            this.attemptMove(this.draggedFrom, targetSquareId);
        }
        
        this.clearDragState();
    }

    handleDragEnd(e) {
        this.clearDragState();
    }

    clearDragState() {
        if (this.draggedPiece) {
            this.draggedPiece.classList.remove('dragging');
        }
        this.draggedPiece = null;
        this.draggedFrom = null;
        this.clearHighlights();
    }

    canMovePiece(piece) {
        if (!this.gameState.gameActive) return false;
        
        const isWhitePiece = piece.dataset.piece === piece.dataset.piece.toUpperCase();
        const isHumanTurn = this.gameState.currentTurn === this.gameState.humanColor;
        const isHumanPiece = (this.gameState.humanColor === 'white' && isWhitePiece) ||
                           (this.gameState.humanColor === 'black' && !isWhitePiece);
        
        return isHumanTurn && isHumanPiece;
    }

    selectSquare(squareId) {
        this.clearSelection();
        this.selectedSquare = squareId;
        document.getElementById(squareId).classList.add('highlighted');
        this.showLegalMoves(squareId);
    }

    clearSelection() {
        if (this.selectedSquare) {
            document.getElementById(this.selectedSquare).classList.remove('highlighted');
            this.selectedSquare = null;
        }
        this.clearHighlights();
    }

    showLegalMoves(fromSquare) {
        // This will be populated with actual legal moves from the backend
        // For now, showing placeholder behavior
        this.clearHighlights();
        
        // Fetch legal moves from backend
        fetch('/api/legal_moves')
            .then(response => response.json())
            .then(data => {
                if (data.legal_moves) {
                    data.legal_moves.forEach(move => {
                        if (move.startsWith(fromSquare)) {
                            const toSquare = move.substring(2, 4);
                            const targetSquare = document.getElementById(toSquare);
                            if (targetSquare) {
                                const hasPiece = targetSquare.querySelector('.piece');
                                targetSquare.classList.add(hasPiece ? 'legal-capture' : 'legal-move');
                            }
                        }
                    });
                }
            })
            .catch(error => console.log('Could not fetch legal moves'));
    }

    clearHighlights() {
        document.querySelectorAll('.legal-move, .legal-capture').forEach(square => {
            square.classList.remove('legal-move', 'legal-capture');
        });
    }

    attemptMove(from, to) {
        // Check if this is a promotion move
        if (this.isPromotionMove(from, to)) {
            this.showPromotionDialog(from, to);
            return;
        }
        
        const move = from + to;
        this.clearSelection();
        this.makeMove(move);
    }

    isPromotionMove(from, to) {
        const fromSquare = document.getElementById(from);
        const piece = fromSquare?.querySelector('.piece');
        
        if (!piece) return false;
        
        const isPawn = piece.dataset.piece.toLowerCase() === 'p';
        const toRank = parseInt(to[1]);
        
        return isPawn && (toRank === 8 || toRank === 1);
    }

    showPromotionDialog(from, to) {
        const fromSquare = document.getElementById(from);
        const piece = fromSquare?.querySelector('.piece');
        const isWhitePawn = piece.dataset.piece === 'P';
        
        const overlay = document.createElement('div');
        overlay.className = 'promotion-overlay';
        overlay.innerHTML = `
            <div class="promotion-dialog">
                <h3>Choose promotion piece:</h3>
                <div class="promotion-pieces">
                    <div class="promotion-piece" data-piece="q" style="background-image: url('/static/images/${isWhitePawn ? 'w' : 'b'}q.png')" title="Queen"></div>
                    <div class="promotion-piece" data-piece="r" style="background-image: url('/static/images/${isWhitePawn ? 'w' : 'b'}r.png')" title="Rook"></div>
                    <div class="promotion-piece" data-piece="b" style="background-image: url('/static/images/${isWhitePawn ? 'w' : 'b'}b.png')" title="Bishop"></div>
                    <div class="promotion-piece" data-piece="n" style="background-image: url('/static/images/${isWhitePawn ? 'w' : 'b'}n.png')" title="Knight"></div>
                </div>
            </div>
        `;
        
        overlay.addEventListener('click', (e) => {
            if (e.target.classList.contains('promotion-piece')) {
                const promotionPiece = e.target.dataset.piece;
                const move = from + to + promotionPiece;
                overlay.remove();
                this.clearSelection();
                this.makeMove(move);
            } else if (e.target.classList.contains('promotion-overlay')) {
                overlay.remove();
                this.clearSelection();
            }
        });
        
        document.body.appendChild(overlay);
    }

    makeMove(move) {
        fetch('/api/make_move', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ move: move })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                this.gameState.fen = data.board_fen;
                this.updateBoardFromFEN(data.board_fen);
                this.updateGameStatus();
                
                this.showMessage(`Your move: ${move}`, 'success');
                
                if (data.is_game_over) {
                    let gameOverMsg = `Game Over: ${data.game_result}`;
                    if (data.learning_message) {
                        gameOverMsg += ` ${data.learning_message}`;
                    }
                    this.showMessage(gameOverMsg, 'info');
                    this.gameState.gameActive = false;
                } else if (data.ai_move) {
                    // Animate AI move
                    setTimeout(() => {
                        this.gameState.fen = data.board_fen;
                        this.updateBoardFromFEN(data.board_fen);
                        this.showMessage(`AI played: ${data.ai_move}`, 'info');
                        this.updateGameStatus();
                        
                        if (data.is_game_over) {
                            let gameOverMsg = `Game Over: ${data.game_result}`;
                            if (data.learning_message) {
                                gameOverMsg += ` ${data.learning_message}`;
                            }
                            this.showMessage(gameOverMsg, 'info');
                            this.gameState.gameActive = false;
                        }
                    }, 500);
                }
            } else {
                this.showMessage(data.error || 'Invalid move', 'error');
            }
        })
        .catch(error => {
            this.showMessage('Error making move: ' + error.message, 'error');
        });
    }

    showMessage(message, type = 'info') {
        const messageArea = document.getElementById('message-area');
        const messageEl = document.createElement('div');
        messageEl.className = `message ${type}`;
        messageEl.textContent = message;
        
        messageArea.appendChild(messageEl);
        
        setTimeout(() => {
            messageEl.remove();
        }, 4000);
    }

    updateGameStatus() {
        fetch('/api/game_status')
            .then(response => response.json())
            .then(data => {
                document.getElementById('gameStatus').textContent = 
                    data.is_game_over ? (data.game_result || 'Game Over') : 'In Progress';
                document.getElementById('currentTurn').textContent = data.current_player || '-';
                document.getElementById('moveCount').textContent = data.move_count || 0;
                
                this.gameState.currentTurn = data.current_player;
                
                if (data.user_stats) {
                    const stats = data.user_stats;
                    document.getElementById('userGames').textContent = stats.total_games || 0;
                    document.getElementById('aiWinRate').textContent = 
                        ((stats.win_rate || 0) * 100).toFixed(1) + '%';
                    document.getElementById('learningInfo').style.display = 'block';
                }
            });
    }

    startNewGame(color) {
        fetch('/api/new_game', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ color: color })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                this.gameState.humanColor = color;
                this.gameState.gameActive = true;
                this.gameState.fen = data.board_fen;
                this.gameState.currentTurn = 'white';
                
                this.updateBoardFromFEN(data.board_fen);
                this.updateGameStatus();
                
                document.getElementById('learningStatus').textContent = 
                    data.learning_enabled ? '🧠 Enabled' : '❌ Disabled';
                
                this.showMessage(`New game started! You are playing as ${color}.`, 'success');
                
                if (data.ai_move) {
                    setTimeout(() => {
                        this.gameState.fen = data.board_fen;
                        this.updateBoardFromFEN(data.board_fen);
                        this.showMessage(`AI played: ${data.ai_move}`, 'info');
                        this.updateGameStatus();
                    }, 500);
                }
            } else {
                this.showMessage('Failed to start new game', 'error');
            }
        })
        .catch(error => {
            this.showMessage('Error starting game: ' + error.message, 'error');
        });
    }
}

// Initialize the chess board when the page loads
let chessBoard;
document.addEventListener('DOMContentLoaded', function() {
    chessBoard = new ChessBoard();
});