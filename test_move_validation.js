const puppeteer = require('puppeteer');

async function testMoveValidation() {
    const browser = await puppeteer.launch({ headless: false, devtools: true });
    const page = await browser.newPage();
    
    // Listen to console messages
    page.on('console', msg => {
        if (msg.text().includes('Can move piece check') || msg.text().includes('Board flipped')) {
            console.log('🟦 BROWSER:', msg.text());
        }
    });
    
    try {
        console.log('🚀 Testing Move Validation and Board Flipping...\n');
        
        // Navigate to the chess app
        await page.goto('http://127.0.0.1:5000');
        await page.waitForSelector('#chess-board', { timeout: 5000 });
        
        console.log('✅ Page loaded successfully');
        
        // Test 1: Start a new game as White
        console.log('\n🔸 Test 1: Starting new game as White...');
        await page.click('button[onclick*="startNewGame(\'white\')"]');
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Check if pieces are set up correctly for white
        const whitePawnE2 = await page.$('#e2 .piece[data-piece="P"]');
        if (whitePawnE2) {
            console.log('✅ White pieces are on bottom (ranks 1-2)');
        } else {
            console.log('❌ White pieces not positioned correctly');
        }
        
        // Test 2: Try to click and move a white pawn
        console.log('\n🔸 Test 2: Testing white pawn move (e2-e4)...');
        await page.click('#e2');
        await new Promise(resolve => setTimeout(resolve, 500));
        await page.click('#e4');
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Check if move was successful
        const pawnAtE4 = await page.$('#e4 .piece[data-piece="P"]');
        if (pawnAtE4) {
            console.log('✅ White pawn moved successfully to e4');
        } else {
            console.log('❌ White pawn move failed');
        }
        
        // Test 3: Start a new game as Black
        console.log('\n🔸 Test 3: Starting new game as Black...');
        await page.click('button[onclick*="startNewGame(\'black\')"]');
        await new Promise(resolve => setTimeout(resolve, 3000));
        
        // Check if board is flipped for black
        const blackPawnE7 = await page.$('#e7 .piece[data-piece="p"]');
        const isFlipped = await page.evaluate(() => {
            // Check if black pieces are on the bottom rows (visually)
            const e7Square = document.getElementById('e7');
            const e2Square = document.getElementById('e2');
            if (!e7Square || !e2Square) return false;
            
            const e7Rect = e7Square.getBoundingClientRect();
            const e2Rect = e2Square.getBoundingClientRect();
            
            // For black, e7 should be below e2 (higher y value)
            return e7Rect.top > e2Rect.top;
        });
        
        if (isFlipped && blackPawnE7) {
            console.log('✅ Board flipped correctly - Black pieces are now on bottom');
        } else {
            console.log('❌ Board not flipped correctly for black player');
        }
        
        // Test 4: Try to move as black (after AI moves)
        console.log('\n🔸 Test 4: Waiting for AI move, then testing black pawn move...');
        await new Promise(resolve => setTimeout(resolve, 3000)); // Wait for AI to move
        
        // Try to move a black pawn
        await page.click('#e7');
        await new Promise(resolve => setTimeout(resolve, 500));
        await page.click('#e5');
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        const blackPawnAtE5 = await page.$('#e5 .piece[data-piece="p"]');
        if (blackPawnAtE5) {
            console.log('✅ Black pawn moved successfully to e5');
        } else {
            console.log('❌ Black pawn move failed');
        }
        
        console.log('\n📊 Test completed! Check console logs above for detailed results.');
        
        // Keep browser open for manual inspection
        await new Promise(resolve => setTimeout(resolve, 5000));
        
    } catch (error) {
        console.error('❌ Test failed:', error);
    } finally {
        await browser.close();
    }
}

testMoveValidation();