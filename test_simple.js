const puppeteer = require('puppeteer');

async function testSimple() {
    const browser = await puppeteer.launch({ headless: false, devtools: true });
    const page = await browser.newPage();
    
    page.on('console', msg => console.log('🟦 BROWSER:', msg.text()));
    
    await page.goto('http://127.0.0.1:5000');
    await page.waitForSelector('#chess-board');
    
    console.log('✅ Page loaded - start new game as white');
    await page.click('button[onclick*="startNewGame(\'white\')"]');
    
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    console.log('🔍 Checking piece at e2...');
    const pieceAtE2 = await page.evaluate(() => {
        const e2 = document.getElementById('e2');
        const piece = e2?.querySelector('.piece');
        return piece ? {
            piece: piece.dataset.piece,
            exists: true,
            squareId: e2.id
        } : { exists: false, squareId: e2?.id };
    });
    
    console.log('E2 info:', pieceAtE2);
    
    console.log('🔍 Checking all pieces on rank 2...');
    const rank2Pieces = await page.evaluate(() => {
        const squares = ['a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2'];
        return squares.map(sq => {
            const square = document.getElementById(sq);
            const piece = square?.querySelector('.piece');
            return {
                square: sq,
                piece: piece?.dataset.piece,
                exists: !!piece
            };
        });
    });
    
    console.log('Rank 2 pieces:', rank2Pieces);
    
    // Keep browser open for inspection
    await new Promise(resolve => setTimeout(resolve, 30000));
    await browser.close();
}

testSimple();