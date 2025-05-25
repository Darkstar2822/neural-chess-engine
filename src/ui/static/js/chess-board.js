// Chess Board JavaScript with Drag & Drop
class ChessBoard {
    constructor() {
        // Audio support
        this.sounds = {
            move: new Audio('/static/sounds/move.mp3'),
            capture: new Audio('/static/sounds/capture.mp3'),
            check: new Audio('/static/sounds/check.mp3'),
            gameOver: new Audio('/static/sounds/game_over.mp3')
        };
        this.selectedSquare = null;
        this.gameState = {
            fen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
            humanColor: 'white',
            gameActive: true,  // Start active so pieces can be moved
            currentTurn: 'white'
        };
        this.draggedPiece = null;
        this.draggedFrom = null;
        this.lastMove = null; // Track the last move
        
        // Bind event handlers to maintain 'this' context
        this.boundDragStart = this.handleDragStart.bind(this);
        this.boundDragEnd = this.handleDragEnd.bind(this);
        
        this.initializeBoard();
        this.setupEventListeners();
        
        // Attach drag events after board is created
        setTimeout(() => this.attachDragEvents(), 100);
    }

    initializeBoard() {
        const boardElement = document.getElementById('chess-board');
        boardElement.innerHTML = '';

        // Determine ranks and files order based on humanColor
        const ranks = this.gameState.humanColor === 'black'
            ? [...Array(8).keys()]
            : [...Array(8).keys()].reverse();
        const files = this.gameState.humanColor === 'black'
            ? [...Array(8).keys()].reverse()
            : [...Array(8).keys()];

        for (let rankIndex of ranks) {
            for (let fileIndex of files) {
                // Consistent square ID logic with updateBoardFromFEN
                const actualRank = this.gameState.humanColor === 'black'
                    ? rankIndex + 1
                    : 8 - rankIndex;
                const file = fileIndex + 1;
                const squareId = String.fromCharCode(96 + file) + actualRank;
                const square = document.createElement('div');
                square.id = squareId;
                // Compute file and rank for coloring and coordinates
                const rank = actualRank;
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
                square.addEventListener('dragenter', (e) => e.preventDefault());

                boardElement.appendChild(square);
            }
        }

        this.updateBoardFromFEN(this.gameState.fen);
    }

    setupEventListeners() {
        // Event listeners are now attached directly to pieces when created
        // This prevents conflicts and ensures proper event handling
    }

    updateBoardFromFEN(fen) {
        const [position] = fen.split(' ');
        const ranks = position.split('/');
        const renderRanks = this.gameState.humanColor === 'black'
            ? [...ranks]
            : [...ranks].reverse();

        // Clear all pieces
        document.querySelectorAll('.piece').forEach(piece => piece.remove());

        // Remove check highlight from all squares
        document.querySelectorAll('.square.highlight-check').forEach(sq => sq.classList.remove('highlight-check'));

        renderRanks.forEach((rank, rankIndex) => {
            let fileIndex = 0;
            for (let char of rank) {
                if (isNaN(char)) {
                    // For black: rank 0 = rank 8, rank 1 = rank 7, etc.
                    // For white: rank 0 = rank 1, rank 1 = rank 2, etc. (after reversing)
                    const actualRank = this.gameState.humanColor === 'black'
                        ? 8 - rankIndex
                        : rankIndex + 1;
                    const squareId = String.fromCharCode(97 + fileIndex) + actualRank;
                    const square = document.getElementById(squareId);
                    if (square) {
                        const piece = this.createPieceElement(char);
                        if (piece) {
                            piece.classList.add('fade-in');
                            square.appendChild(piece);
                            
                        }
                    }
                    fileIndex++;
                } else {
                    fileIndex += parseInt(char);
                }
            }
        });

        // Highlight the last move squares, if any
        if (this.lastMove) {
            const from = this.lastMove.substring(0, 2);
            const to = this.lastMove.substring(2, 4);
            const fromSquare = document.getElementById(from);
            const toSquare = document.getElementById(to);
            if (fromSquare) fromSquare.classList.add('highlighted');
            if (toSquare) toSquare.classList.add('highlighted');
        }

        // Check for check or checkmate conditions
        const boardState = this.gameState.fen;
        if (boardState.includes('+')) {
            this.sounds.check.play();
            // Find the king in check and highlight its square
            const kingSquare = [...document.querySelectorAll('.square')].find(sq => {
                const piece = sq.querySelector('.piece');
                return piece && piece.dataset.piece.toLowerCase() === 'k';
            });
            if (kingSquare) kingSquare.classList.add('highlight-check');
        }

        // Re-attach drag events after updating board
        this.attachDragEvents();
    }
    
    attachDragEvents() {
        console.log('Attaching drag events to all pieces...');
        const pieces = document.querySelectorAll('.piece');
        pieces.forEach((piece, index) => {
            // Remove existing listeners first
            piece.removeEventListener('dragstart', this.boundDragStart);
            piece.removeEventListener('dragend', this.boundDragEnd);
            
            // Add new listeners
            piece.addEventListener('dragstart', this.boundDragStart);
            piece.addEventListener('dragend', this.boundDragEnd);
            
            console.log(`Attached drag events to piece ${index}: ${piece.dataset.piece}`);
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
        console.log('Drag start triggered', e.target);
        
        if (!e.target.classList.contains('piece')) {
            console.log('Not a piece element');
            return;
        }
        
        const piece = e.target;
        const square = piece.parentElement;
        
        console.log(`Attempting to drag ${piece.dataset.piece} from ${square.id}`);
        
        if (!this.canMovePiece(piece)) {
            console.log('Cannot move this piece');
            e.preventDefault();
            return;
        }
        
        this.draggedPiece = piece;
        this.draggedFrom = square.id;
        piece.classList.add('dragging');
        
        console.log(`Drag started: ${this.draggedFrom}`);
        
        // Show legal moves
        this.showLegalMoves(square.id);
        
        if (e.dataTransfer) {
            e.dataTransfer.effectAllowed = 'move';
            e.dataTransfer.setData('text/plain', '');
        }
    }

    handleDragOver(e) {
        e.preventDefault();
        if (e.dataTransfer) {
            e.dataTransfer.dropEffect = 'move';
        }
    }

    handleDrop(e) {
        e.preventDefault();
        
        if (!this.draggedPiece || !this.draggedFrom) {
            this.clearDragState();
            return;
        }
        
        const targetSquare = e.currentTarget;
        const targetSquareId = targetSquare.id;
        
        console.log(`Drag drop: ${this.draggedFrom} -> ${targetSquareId}`); // Debug log
        
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
        // Snapback logic: return piece to origin if needed
        if (this.draggedFrom && this.draggedPiece) {
            const originSquare = document.getElementById(this.draggedFrom);
            if (originSquare && !originSquare.contains(this.draggedPiece)) {
                originSquare.appendChild(this.draggedPiece);
            }
        }
        this.draggedPiece = null;
        this.draggedFrom = null;
        this.clearHighlights();
    }

    canMovePiece(piece) {
        if (!this.gameState.gameActive) {
            console.log('Game not active');
            return false;
        }
        
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
                                const isEnPassant = this.isEnPassantMove(fromSquare, toSquare);
                                const isCastling = this.isCastlingMove(fromSquare, toSquare);
                                
                                if (isEnPassant) {
                                    targetSquare.classList.add('legal-en-passant');
                                } else if (isCastling) {
                                    targetSquare.classList.add('legal-castling');
                                } else if (hasPiece) {
                                    targetSquare.classList.add('legal-capture');
                                } else {
                                    targetSquare.classList.add('legal-move');
                                }
                            }
                        }
                    });
                }
            })
            .catch(error => console.log('Could not fetch legal moves'));
    }

    clearHighlights() {
        document.querySelectorAll('.legal-move, .legal-capture, .legal-en-passant, .legal-castling, .highlighted').forEach(square => {
            square.classList.remove('legal-move', 'legal-capture', 'legal-en-passant', 'legal-castling', 'highlighted');
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
        const fromRank = parseInt(from[1]);
        
        // White pawn reaching rank 8, or black pawn reaching rank 1
        const isWhitePawn = piece.dataset.piece === 'P';
        const isBlackPawn = piece.dataset.piece === 'p';
        
        return isPawn && ((isWhitePawn && fromRank === 7 && toRank === 8) || 
                         (isBlackPawn && fromRank === 2 && toRank === 1));
    }

    isEnPassantMove(from, to) {
        const fromSquare = document.getElementById(from);
        const toSquare = document.getElementById(to);
        const piece = fromSquare?.querySelector('.piece');
        
        if (!piece || piece.dataset.piece.toLowerCase() !== 'p') return false;
        
        const fromFile = from.charCodeAt(0);
        const toFile = to.charCodeAt(0);
        const fromRank = parseInt(from[1]);
        const toRank = parseInt(to[1]);
        
        // Pawn moving diagonally to empty square (en passant condition)
        const isDiagonal = Math.abs(fromFile - toFile) === 1 && Math.abs(fromRank - toRank) === 1;
        const isEmptyTarget = !toSquare?.querySelector('.piece');
        
        return isDiagonal && isEmptyTarget;
    }

    isCastlingMove(from, to) {
        const fromSquare = document.getElementById(from);
        const piece = fromSquare?.querySelector('.piece');
        
        if (!piece || piece.dataset.piece.toLowerCase() !== 'k') return false;
        
        const fromFile = from.charCodeAt(0);
        const toFile = to.charCodeAt(0);
        
        // King moving 2 squares horizontally
        return Math.abs(fromFile - toFile) === 2;
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
        // Check special moves before making them
        const isEnPassant = move.length === 4 && this.isEnPassantMove(move.substring(0, 2), move.substring(2, 4));
        const isCastling = move.length === 4 && this.isCastlingMove(move.substring(0, 2), move.substring(2, 4));

        fetch('/api/make_move', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ move: move })
        })
        .then(response => response.json())
        .then(data => {
            // Play game over sound if game is over at the start
            if (data.is_game_over) {
                this.sounds.gameOver.play();
            }
            if (data.status === 'success') {
                // Play move or capture sound (before updating FEN)
                const isCapture = document.getElementById(move.substring(2, 4)).querySelector('.piece');
                if (isCapture) {
                    this.sounds.capture.play();
                } else {
                    this.sounds.move.play();
                }
                // Track the last move (player)
                this.lastMove = move.substring(0, 4);
                this.gameState.fen = data.board_fen;
                this.updateBoardFromFEN(data.board_fen);
                
                // Update current turn after move
                this.updateGameStatus();
                
                let moveMessage = `Your move: ${move}`;
                if (isEnPassant) {
                    moveMessage += ' (en passant)';
                } else if (isCastling) {
                    const isKingside = move.substring(2, 4) === 'g1' || move.substring(2, 4) === 'g8';
                    moveMessage += isKingside ? ' (kingside castling)' : ' (queenside castling)';
                }
                this.showMessage(moveMessage, 'success');
                
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
                        // Track the last move (AI)
                        this.lastMove = data.ai_move?.substring(0, 4);
                        this.gameState.fen = data.board_fen;
                        this.updateBoardFromFEN(data.board_fen);
                        this.showMessage(`AI played: ${data.ai_move}`, 'info');
                        this.updateGameStatus();
                        // Check detection after AI move
                        if (data.board_fen.includes('k') && data.board_fen.includes('K')) {
                            // Check for check symbol in FEN
                            if (data.board_fen.includes('+')) {
                                this.sounds.check.play();
                            } else {
                                this.sounds.move.play();
                            }
                        }
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
            // Snapback logic: return piece to origin if needed
            if (this.draggedFrom && this.draggedPiece) {
                const originSquare = document.getElementById(this.draggedFrom);
                if (originSquare && !originSquare.contains(this.draggedPiece)) {
                    originSquare.appendChild(this.draggedPiece);
                }
            }
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

                // Rebuild board with correct orientation and FEN - this flips the board automatically
                this.initializeBoard();
                this.updateGameStatus();
                
                console.log(`Board flipped for ${color} player - human pieces now on bottom`);

                document.getElementById('learningStatus').textContent =
                    data.learning_enabled ? '🧠 Enabled' : '❌ Disabled';

                this.showMessage(`New game started! You are playing as ${color}.`, 'success');

                if (data.ai_move) {
                    setTimeout(() => {
                        this.gameState.fen = data.board_fen;
                        this.initializeBoard();
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