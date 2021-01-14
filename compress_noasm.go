// +build !amd64

package blake3

import "encoding/binary"

func compressNode(n Node) (out [16]uint32) {
	CompressNodeGeneric(&out, n)
	return
}

func compressBuffer(buf *[maxSIMD * chunkSize]byte, buflen int, key *[8]uint32, Counter uint64, Flags uint32) Node {
	return compressBufferGeneric(buf, buflen, key, Counter, Flags)
}

func compressChunk(chunk []byte, key *[8]uint32, Counter uint64, Flags uint32) Node {
	n := Node{
		CV:       *key,
		Counter:  Counter,
		BlockLen: blockSize,
		Flags:    Flags | flagChunkStart,
	}
	var Block [blockSize]byte
	for len(chunk) > blockSize {
		copy(Block[:], chunk)
		chunk = chunk[blockSize:]
		bytesToWords(Block, &n.Block)
		n.CV = chainingValue(n)
		n.Flags &^= flagChunkStart
	}
	// pad last Block with zeros
	Block = [blockSize]byte{}
	n.BlockLen = uint32(len(chunk))
	copy(Block[:], chunk)
	bytesToWords(Block, &n.Block)
	n.Flags |= flagChunkEnd
	return n
}

func hashBlock(out *[64]byte, buf []byte) {
	var Block [64]byte
	var words [16]uint32
	copy(Block[:], buf)
	bytesToWords(Block, &words)
	CompressNodeGeneric(&words, Node{
		CV:       iv,
		Block:    words,
		BlockLen: uint32(len(buf)),
		Flags:    flagChunkStart | flagChunkEnd | flagRoot,
	})
	wordsToBytes(words, out)
}

func compressBlocks(out *[maxSIMD * blockSize]byte, n Node) {
	var outs [maxSIMD][64]byte
	compressBlocksGeneric(&outs, n)
	for i := range outs {
		copy(out[i*64:], outs[i][:])
	}
}

func mergeSubtrees(cvs *[maxSIMD][8]uint32, numCVs uint64, key *[8]uint32, Flags uint32) Node {
	return mergeSubtreesGeneric(cvs, numCVs, key, Flags)
}

func bytesToWords(bytes [64]byte, words *[16]uint32) {
	for i := range words {
		words[i] = binary.LittleEndian.Uint32(bytes[4*i:])
	}
}

func wordsToBytes(words [16]uint32, Block *[64]byte) {
	for i, w := range words {
		binary.LittleEndian.PutUint32(Block[4*i:], w)
	}
}
