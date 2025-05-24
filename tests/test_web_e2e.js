/**
 * End-to-end tests for the Chess Web Interface using Puppeteer
 */

const puppeteer = require('puppeteer');
const { spawn } = require('child_process');
const path = require('path');

describe('Chess Web Interface E2E Tests', () => {
    let browser;
    let page;
    let server;
    const BASE_URL = 'http://localhost:5000';
    
    beforeAll(async () => {
        // Start the Flask development server
        console.log('Starting Flask server...');
        server = spawn('python', ['main.py', 'web', '--no-learn'], {
            cwd: path.join(__dirname, '..'),
            stdio: 'pipe'
        });
        
        // Wait for server to start
        await new Promise((resolve) => {
            server.stdout.on('data', (data) => {
                if (data.toString().includes('Running on')) {
                    console.log('Flask server started');
                    resolve();
                }
            });
            
            server.stderr.on('data', (data) => {
                console.log('Server stderr:', data.toString());
            });
        });
        
        // Wait a bit more to ensure server is ready
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Launch browser
        browser = await puppeteer.launch({
            headless: process.env.CI === 'true',
            slowMo: 100,
            devtools: false
        });
        
        page = await browser.newPage();
        page.setDefaultTimeout(10000);
        
        // Set viewport
        await page.setViewport({ width: 1280, height: 720 });
        
        // Listen for console logs and errors
        page.on('console', msg => {
            if (msg.type() === 'error') {
                console.log('Browser console error:', msg.text());
            }
        });
        
        page.on('pageerror', err => {
            console.log('Page error:', err.message);
        });
    });
    
    afterAll(async () => {
        if (browser) {
            await browser.close();
        }
        if (server) {
            server.kill('SIGTERM');
        }
    });
    
    beforeEach(async () => {
        await page.goto(BASE_URL, { waitUntil: 'networkidle0' });
    });
    
    test('Page loads correctly', async () => {
        await page.goto(BASE_URL);
        
        // Check page title
        const title = await page.title();
        expect(title).toContain('Neural Chess Engine');
        
        // Check main elements exist
        await page.waitForSelector('#chess-board');
        await page.waitForSelector('.controls-panel');
        
        // Check that chess board has 64 squares
        const squares = await page.$$('.square');
        expect(squares.length).toBe(64);
    });
    
    test('Chess pieces are loaded and visible', async () => {
        await page.goto(BASE_URL);
        
        // Wait for board to load
        await page.waitForSelector('#chess-board');
        
        // Check that pieces are present (should be 32 in starting position)
        const pieces = await page.$$('.piece');
        expect(pieces.length).toBe(32);
        
        // Check specific pieces exist
        const whiteKing = await page.$('[data-piece="K"]');
        const blackKing = await page.$('[data-piece="k"]');
        expect(whiteKing).toBeTruthy();
        expect(blackKing).toBeTruthy();
    });
    
    test('Start new game as white', async () => {
        await page.goto(BASE_URL);
        
        // Click "Play as White" button
        await page.click('button:contains("Play as White")');
        
        // Wait for game to initialize
        await page.waitForTimeout(1000);
        
        // Check game status
        const status = await page.$eval('#gameStatus', el => el.textContent);
        expect(status).toContain('active');
        
        // Check current turn
        const turn = await page.$eval('#currentTurn', el => el.textContent);
        expect(turn).toBe('White');
    });
    
    test('Start new game as black', async () => {
        await page.goto(BASE_URL);
        
        // Click "Play as Black" button  
        await page.click('button:contains("Play as Black")');
        
        // Wait for AI to make first move
        await page.waitForTimeout(2000);
        
        // Check that AI made a move (turn should be black now)
        const turn = await page.$eval('#currentTurn', el => el.textContent);
        expect(turn).toBe('Black');
        
        // Check move count increased
        const moveCount = await page.$eval('#moveCount', el => el.textContent);
        expect(parseInt(moveCount)).toBeGreaterThan(0);
    });
    
    test('Make move via manual input', async () => {
        await page.goto(BASE_URL);
        
        // Start game as white
        await page.click('button:contains("Play as White")');
        await page.waitForTimeout(1000);
        
        // Enter move in manual input
        await page.type('#manualMoveInput', 'e2e4');
        await page.click('button:contains("Make Move")');
        
        // Wait for move to process
        await page.waitForTimeout(1500);
        
        // Check that move was made (turn should change)
        const turn = await page.$eval('#currentTurn', el => el.textContent);
        expect(turn).toBe('White'); // Should be white's turn again after AI response
        
        // Check move count increased
        const moveCount = await page.$eval('#moveCount', el => el.textContent);
        expect(parseInt(moveCount)).toBeGreaterThan(1);
    });
    
    test('Click to select and move pieces', async () => {
        await page.goto(BASE_URL);
        
        // Start game as white
        await page.click('button:contains("Play as White")');
        await page.waitForTimeout(1000);
        
        // Click on white pawn (e2 square)
        await page.click('#e2');
        
        // Check if square is highlighted
        const e2Square = await page.$('#e2.highlighted');
        expect(e2Square).toBeTruthy();
        
        // Click destination square (e4)
        await page.click('#e4');
        
        // Wait for move to process
        await page.waitForTimeout(1500);
        
        // Check move was made - pawn should no longer be on e2
        const e2Piece = await page.$('#e2 .piece');
        expect(e2Piece).toBeFalsy();
        
        // Check pawn is now on e4
        const e4Piece = await page.$('#e4 .piece');
        expect(e4Piece).toBeTruthy();
    });
    
    test('Drag and drop functionality', async () => {
        await page.goto(BASE_URL);
        
        // Start game as white
        await page.click('button:contains("Play as White")');
        await page.waitForTimeout(1000);
        
        // Test drag and drop
        const pawnE2 = await page.$('#e2 .piece');
        const squareE4 = await page.$('#e4');
        
        if (pawnE2 && squareE4) {
            // Get bounding boxes
            const pawnBox = await pawnE2.boundingBox();
            const squareBox = await squareE4.boundingBox();
            
            // Perform drag and drop
            await page.mouse.move(pawnBox.x + pawnBox.width/2, pawnBox.y + pawnBox.height/2);
            await page.mouse.down();
            await page.mouse.move(squareBox.x + squareBox.width/2, squareBox.y + squareBox.height/2);
            await page.mouse.up();
            
            // Wait for move to process
            await page.waitForTimeout(1500);
            
            // Check move was made
            const moveCount = await page.$eval('#moveCount', el => el.textContent);
            expect(parseInt(moveCount)).toBeGreaterThan(1);
        }
    });
    
    test('Invalid move handling', async () => {
        await page.goto(BASE_URL);
        
        // Start game as white
        await page.click('button:contains("Play as White")');
        await page.waitForTimeout(1000);
        
        // Try invalid move
        await page.type('#manualMoveInput', 'e2e5'); // Invalid pawn move
        await page.click('button:contains("Make Move")');
        
        // Wait for error message
        await page.waitForTimeout(1000);
        
        // Check for error message (should appear in message area)
        const messageArea = await page.$('#message-area');
        if (messageArea) {
            const messageText = await page.evaluate(el => el.textContent, messageArea);
            expect(messageText).toContain('Invalid move');
        }
    });
    
    test('Legal move indicators', async () => {
        await page.goto(BASE_URL);
        
        // Start game as white
        await page.click('button:contains("Play as White")');
        await page.waitForTimeout(1000);
        
        // Click on a piece to select it
        await page.click('#e2');
        
        // Check for legal move indicators
        await page.waitForTimeout(500);
        const legalMoveSquares = await page.$$('.legal-move');
        expect(legalMoveSquares.length).toBeGreaterThan(0);
    });
    
    test('Game status updates correctly', async () => {
        await page.goto(BASE_URL);
        
        // Start game
        await page.click('button:contains("Play as White")');
        await page.waitForTimeout(1000);
        
        // Make a move
        await page.type('#manualMoveInput', 'e2e4');
        await page.click('button:contains("Make Move")');
        await page.waitForTimeout(1500);
        
        // Check status updates
        const status = await page.$eval('#gameStatus', el => el.textContent);
        expect(status).toBeTruthy();
        
        const moveCount = await page.$eval('#moveCount', el => el.textContent);
        expect(parseInt(moveCount)).toBeGreaterThan(0);
    });
    
    test('Reset game functionality', async () => {
        await page.goto(BASE_URL);
        
        // Start game and make some moves
        await page.click('button:contains("Play as White")');
        await page.waitForTimeout(1000);
        
        await page.type('#manualMoveInput', 'e2e4');
        await page.click('button:contains("Make Move")');
        await page.waitForTimeout(1500);
        
        // Reset game
        await page.click('button:contains("Reset Game")');
        await page.waitForTimeout(2000);
        
        // Check game was reset (move count should be 0)
        const moveCount = await page.$eval('#moveCount', el => el.textContent);
        expect(parseInt(moveCount)).toBe(0);
    });
    
    test('Test drag & drop button functionality', async () => {
        await page.goto(BASE_URL);
        
        // Click test button
        await page.click('button:contains("Test Drag & Drop")');
        
        // Check console output (this is for debugging)
        // The test button should log information about drag & drop state
        await page.waitForTimeout(1000);
    });
    
    test('Chess piece images load correctly', async () => {
        await page.goto(BASE_URL);
        
        // Wait for images to load
        await page.waitForTimeout(2000);
        
        // Check that piece images are not broken
        const brokenImages = await page.evaluate(() => {
            const images = Array.from(document.querySelectorAll('img'));
            return images.filter(img => !img.complete || img.naturalWidth === 0).length;
        });
        
        expect(brokenImages).toBe(0);
    });
    
    test('Responsive design - mobile viewport', async () => {
        // Test mobile viewport
        await page.setViewport({ width: 375, height: 667 });
        await page.goto(BASE_URL);
        
        // Check that board is still visible and usable
        const board = await page.$('#chess-board');
        expect(board).toBeTruthy();
        
        const boardBox = await board.boundingBox();
        expect(boardBox.width).toBeGreaterThan(0);
        expect(boardBox.height).toBeGreaterThan(0);
    });
});

// Helper to check if element contains text (since Puppeteer doesn't have native :contains)
async function clickButtonWithText(page, text) {
    await page.evaluate((text) => {
        const buttons = Array.from(document.querySelectorAll('button'));
        const button = buttons.find(btn => btn.textContent.includes(text));
        if (button) button.click();
    }, text);
}