import { createWriteStream } from 'node:fs';
import { pipeline } from 'node:stream/promises';
import { createCanvas } from 'canvas';
import { determineVersion, errorCorrectionCodewords, getTotalDataBits, pxDimensions } from './constants.js';

const GF256_EXP = new Uint8Array(512);
const GF256_LOG = new Uint8Array(256);

(function initGF256() {
	let x = 1;
	for (let idx = 0; idx < 255; idx++) {
		GF256_EXP[idx] = x;
		GF256_LOG[x] = idx;
		x = (x << 1) ^ (x & 0x80 ? 0x11d : 0);
	}

	for (let idx = 255; idx < 512; idx++) {
		GF256_EXP[idx] = GF256_EXP[idx - 255]!;
	}
})();

function multiplyGF256(x: number, y: number) {
	if (x === 0 || y === 0) return 0;
	return GF256_EXP[GF256_LOG[x]! + GF256_LOG[y]!]!;
}

function reedSolomonGenerator(degree: number) {
	const coefficients = new Uint8Array(degree + 1);
	coefficients[0] = 1;

	for (let deg = 0; deg < degree; deg++) {
		for (let coeffIdx = degree; coeffIdx > 0; coeffIdx--) {
			coefficients[coeffIdx] = coefficients[coeffIdx - 1]! ^ multiplyGF256(coefficients[coeffIdx]!, GF256_EXP[deg]!);
		}

		coefficients[0] = multiplyGF256(coefficients[0], GF256_EXP[deg]!);
	}

	return coefficients;
}

function reedSolomonEncode(data: Uint8Array, degree: number) {
	const coefficients = reedSolomonGenerator(degree);
	const result = new Uint8Array(data.length + degree);
	result.set(data);

	for (let idx = 0; idx < data.length; idx++) {
		const coef = result[idx];
		if (coef !== 0) {
			for (const [jdx, coefficient] of coefficients.entries()) {
				result[idx + jdx]! ^= multiplyGF256(coef!, coefficient);
			}
		}
	}

	return result.slice(data.length);
}

const formatBits = [1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1];

function createQRCode(dataBits: (0 | 1)[], version: number) {
	const size = pxDimensions[version as keyof typeof pxDimensions];
	const canvas = createCanvas(size * 4, size * 4); // Scale up for better scanning
	const ctx = canvas.getContext('2d');

	ctx.fillStyle = 'white';
	ctx.fillRect(0, 0, size * 4, size * 4);

	const modules = Array.from({ length: size }, () => Array.from({ length: size }, () => false));
	const reserved = Array.from({ length: size }, () => Array.from({ length: size }, () => false));

	function setModule(x: number, y: number, value: boolean, isReserved = true) {
		if (x >= 0 && x < size && y >= 0 && y < size && !reserved[y]![x]) {
			modules[y]![x] = value;
			if (isReserved) {
				reserved[y]![x] = true;
			}

			ctx.fillStyle = value ? 'black' : 'white';
			ctx.fillRect(x * 4, y * 4, 4, 4);
		}
	}

	function drawFinderPattern(x: number, y: number) {
		for (let dy = -1; dy <= 7; dy++) {
			for (let dx = -1; dx <= 7; dx++) {
				if (dx >= 0 && dx <= 6 && dy >= 0 && dy <= 6) {
					const value = dx === 0 || dx === 6 || dy === 0 || dy === 6 || (dx >= 2 && dx <= 4 && dy >= 2 && dy <= 4);
					setModule(x + dx, y + dy, value);
				} else if (dx >= -1 && dx <= 7 && dy >= -1 && dy <= 7) {
					setModule(x + dx, y + dy, false);
				}
			}
		}
	}

	drawFinderPattern(0, 0);
	drawFinderPattern(size - 7, 0);
	drawFinderPattern(0, size - 7);

	for (let idx = 8; idx < size - 8; idx++) {
		const value = idx % 2 === 0;
		setModule(idx, 6, value);
		setModule(6, idx, value);
	}

	for (let idx = 0; idx < 15; idx++) {
		const bit = formatBits[idx];

		if (idx < 6) {
			setModule(idx, 8, bit === 1);
		} else if (idx < 8) {
			setModule(idx + 1, 8, bit === 1);
		} else if (idx < 9) {
			setModule(8, 7, bit === 1);
		} else {
			setModule(8, 14 - idx, bit === 1);
		}

		if (idx < 8) {
			setModule(size - 1 - idx, 8, bit === 1);
		} else {
			setModule(8, size - 15 + idx, bit === 1);
		}
	}

	function drawAlignmentPattern(x: number, y: number) {
		for (let dy = -2; dy <= 2; dy++) {
			for (let dx = -2; dx <= 2; dx++) {
				const isPartOfPattern = Math.abs(dx) === 2 || Math.abs(dy) === 2 || (dx === 0 && dy === 0);
				setModule(x + dx, y + dy, isPartOfPattern);
			}
		}
	}

	if (version >= 2) {
		const positions = [6, size - 7];
		for (const row of positions) {
			for (const col of positions) {
				if ((row === 6 && col === 6) || (row === 6 && col === size - 7) || (row === size - 7 && col === 6)) {
					continue;
				}

				drawAlignmentPattern(row, col);
			}
		}
	}

	// dark
	setModule(size - 8, 8, true);

	let bitIndex = 0;
	for (let right = size - 1; right >= 1; right -= 2) {
		if (right === 6) {
			right = 5; // skip timing
		}

		const upward = ((size - right) & 2) === 0;
		for (let vert = 0; vert < size; vert++) {
			const y = upward ? size - 1 - vert : vert;
			for (const x of [right, right - 1]) {
				if (x >= 0 && !reserved[y]![x] && bitIndex < dataBits.length) {
					let bit = dataBits[bitIndex++] === 1;

					if ((y + x) % 2 === 0) {
						bit = !bit;
					}

					setModule(x, y, bit, false);
				}
			}
		}
	}

	const quietZone = createCanvas((size + 8) * 4, (size + 8) * 4);
	const qzCtx = quietZone.getContext('2d');
	qzCtx.fillStyle = 'white';
	qzCtx.fillRect(0, 0, (size + 8) * 4, (size + 8) * 4);
	qzCtx.drawImage(canvas, 16, 16);

	return quietZone;
}

async function saveQRCode(input: string, path: string) {
	const version = determineVersion(input);
	const dataBits: (0 | 1)[] = [];

	// byte mode
	dataBits.push(0, 1, 0, 0);

	// character count indicator
	const byteLength = Buffer.byteLength(input, 'utf8');
	const countBits = version <= 9 ? 8 : 16;
	const lengthBits = byteLength.toString(2).padStart(countBits, '0').split('').map(Number) as (0 | 1)[];
	dataBits.push(...lengthBits);

	// Data
	const bytes = Buffer.from(input, 'utf8');
	for (const byte of bytes) {
		for (let idx = 7; idx >= 0; idx--) {
			dataBits.push(((byte >> idx) & 1) as 0 | 1);
		}
	}

	const totalBits = getTotalDataBits(version);
	const terminatorLength = Math.min(4, totalBits - dataBits.length);
	dataBits.push(...Array.from({ length: terminatorLength }, (): 0 | 1 => 0));

	// pad to byte boundary
	while (dataBits.length % 8 !== 0) {
		dataBits.push(0);
	}

	// pad bytes
	const padBytes = [0b11101100, 0b00010001];
	while (dataBits.length < totalBits) {
		dataBits.push(
			...(padBytes[(dataBits.length / 8) % 2]!.toString(2).padStart(8, '0').split('').map(Number) as (0 | 1)[]),
		);
	}

	// convert to bytes for error correction
	const dataBytes = new Uint8Array(dataBits.length / 8);
	for (let idx = 0; idx < dataBytes.length; idx++) {
		dataBytes[idx] = Number.parseInt(dataBits.slice(idx * 8, (idx + 1) * 8).join(''), 2);
	}

	const ecCodewords = errorCorrectionCodewords[version as keyof typeof errorCorrectionCodewords];
	const ecBytes = reedSolomonEncode(dataBytes, ecCodewords);

	// final bit sequence
	const finalBits: (0 | 1)[] = [];
	for (const byte of [...dataBytes, ...ecBytes]) {
		for (let idx = 7; idx >= 0; idx--) {
			finalBits.push(((byte >> idx) & 1) as 0 | 1);
		}
	}

	const qrCodeCanvas = createQRCode(finalBits, version);
	const outStream = createWriteStream(path);
	await pipeline(qrCodeCanvas.createPNGStream(), outStream);
}

await saveQRCode('https://cs.unibuc.ro/~crusu/asc/index.html', 'qrCode1.png');
await saveQRCode('insert team name here', 'qrCode2.png');
