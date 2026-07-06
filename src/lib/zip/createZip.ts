/**
 * Minimal ZIP writer (STORE method — files embedded uncompressed, no
 * deflate) for bundling batch-export results into one download. Written by
 * hand rather than pulling in a dependency (e.g. jszip) since the format
 * itself is simple and this project otherwise avoids dependencies for
 * things a few dozen lines of code can do directly.
 *
 * Format reference: a .zip is [local file header + data]... repeated, then
 * [central directory entry]... mirroring each file, then a single
 * end-of-central-directory record. All multi-byte fields are little-endian.
 */

interface ZipEntry {
  name: string;
  data: Uint8Array;
}

let crcTable: Uint32Array | null = null;
function getCrcTable(): Uint32Array {
  if (crcTable) return crcTable;
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[n] = c >>> 0;
  }
  crcTable = table;
  return table;
}

function crc32(data: Uint8Array): number {
  const table = getCrcTable();
  let crc = 0xffffffff;
  for (let i = 0; i < data.length; i++) {
    crc = table[(crc ^ data[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

// MS-DOS date/time packing — the exact timestamp doesn't matter for this
// use case, so this just stamps every entry with the moment the zip is built.
function dosDateTime(date: Date): { time: number; date: number } {
  const time = (date.getHours() << 11) | (date.getMinutes() << 5) | (date.getSeconds() >> 1);
  const dosDate = ((date.getFullYear() - 1980) << 9) | ((date.getMonth() + 1) << 5) | date.getDate();
  return { time, date: dosDate };
}

function writeUint32LE(view: DataView, offset: number, value: number) {
  view.setUint32(offset, value, true);
}
function writeUint16LE(view: DataView, offset: number, value: number) {
  view.setUint16(offset, value, true);
}

export function createZip(entries: ZipEntry[]): Blob {
  const encoder = new TextEncoder();
  const { time, date } = dosDateTime(new Date());
  const chunks: Uint8Array[] = [];
  const centralDirectoryChunks: Uint8Array[] = [];
  let offset = 0;

  for (const entry of entries) {
    const nameBytes = encoder.encode(entry.name);
    const crc = crc32(entry.data);
    const size = entry.data.length;

    const localHeader = new Uint8Array(30 + nameBytes.length);
    const localView = new DataView(localHeader.buffer);
    writeUint32LE(localView, 0, 0x04034b50);
    writeUint16LE(localView, 4, 20); // version needed
    writeUint16LE(localView, 6, 0); // flags
    writeUint16LE(localView, 8, 0); // method: store
    writeUint16LE(localView, 10, time);
    writeUint16LE(localView, 12, date);
    writeUint32LE(localView, 14, crc);
    writeUint32LE(localView, 18, size); // compressed size
    writeUint32LE(localView, 22, size); // uncompressed size
    writeUint16LE(localView, 26, nameBytes.length);
    writeUint16LE(localView, 28, 0); // extra field length
    localHeader.set(nameBytes, 30);

    chunks.push(localHeader, entry.data);

    const centralEntry = new Uint8Array(46 + nameBytes.length);
    const centralView = new DataView(centralEntry.buffer);
    writeUint32LE(centralView, 0, 0x02014b50);
    writeUint16LE(centralView, 4, 20); // version made by
    writeUint16LE(centralView, 6, 20); // version needed
    writeUint16LE(centralView, 8, 0); // flags
    writeUint16LE(centralView, 10, 0); // method
    writeUint16LE(centralView, 12, time);
    writeUint16LE(centralView, 14, date);
    writeUint32LE(centralView, 16, crc);
    writeUint32LE(centralView, 20, size);
    writeUint32LE(centralView, 24, size);
    writeUint16LE(centralView, 28, nameBytes.length);
    writeUint16LE(centralView, 30, 0); // extra length
    writeUint16LE(centralView, 32, 0); // comment length
    writeUint16LE(centralView, 34, 0); // disk number start
    writeUint16LE(centralView, 36, 0); // internal attrs
    writeUint32LE(centralView, 38, 0); // external attrs
    writeUint32LE(centralView, 42, offset); // local header offset
    centralEntry.set(nameBytes, 46);
    centralDirectoryChunks.push(centralEntry);

    offset += localHeader.length + entry.data.length;
  }

  const centralDirectorySize = centralDirectoryChunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const centralDirectoryOffset = offset;

  const endRecord = new Uint8Array(22);
  const endView = new DataView(endRecord.buffer);
  writeUint32LE(endView, 0, 0x06054b50);
  writeUint16LE(endView, 4, 0); // disk number
  writeUint16LE(endView, 6, 0); // disk with central directory
  writeUint16LE(endView, 8, entries.length); // entries on this disk
  writeUint16LE(endView, 10, entries.length); // total entries
  writeUint32LE(endView, 12, centralDirectorySize);
  writeUint32LE(endView, 16, centralDirectoryOffset);
  writeUint16LE(endView, 20, 0); // comment length

  return new Blob([...chunks, ...centralDirectoryChunks, endRecord], { type: "application/zip" });
}
