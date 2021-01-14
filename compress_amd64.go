package blake3

import "unsafe"

//go:generate go run avo/gen.go -out blake3_amd64.s

//go:noescape
func compressChunksAVX512(cvs *[16][8]uint32, buf *[16 * chunkSize]byte, key *[8]uint32, Counter uint64, Flags uint32)

//go:noescape
func compressChunksAVX2(cvs *[8][8]uint32, buf *[8 * chunkSize]byte, key *[8]uint32, Counter uint64, Flags uint32)

//go:noescape
func compressBlocksAVX512(out *[1024]byte, Block *[16]uint32, CV *[8]uint32, Counter uint64, BlockLen uint32, Flags uint32)

//go:noescape
func compressBlocksAVX2(out *[512]byte, msgs *[16]uint32, CV *[8]uint32, Counter uint64, BlockLen uint32, Flags uint32)

//go:noescape
func compressParentsAVX2(parents *[8][8]uint32, cvs *[16][8]uint32, key *[8]uint32, Flags uint32)

func compressNode(n Node) (out [16]uint32) {
	CompressNodeGeneric(&out, n)
	return
}

func compressBufferAVX512(buf *[maxSIMD * chunkSize]byte, buflen int, key *[8]uint32, Counter uint64, Flags uint32) Node {
	var cvs [maxSIMD][8]uint32
	compressChunksAVX512(&cvs, buf, key, Counter, Flags)
	numChunks := uint64(buflen / chunkSize)
	if buflen%chunkSize != 0 {
		// use non-asm for remainder
		partialChunk := buf[buflen-buflen%chunkSize : buflen]
		cvs[numChunks] = chainingValue(compressChunk(partialChunk, key, Counter+numChunks, Flags))
		numChunks++
	}
	return mergeSubtrees(&cvs, numChunks, key, Flags)
}

func compressBufferAVX2(buf *[maxSIMD * chunkSize]byte, buflen int, key *[8]uint32, Counter uint64, Flags uint32) Node {
	var cvs [maxSIMD][8]uint32
	cvHalves := (*[2][8][8]uint32)(unsafe.Pointer(&cvs))
	bufHalves := (*[2][8 * chunkSize]byte)(unsafe.Pointer(buf))
	compressChunksAVX2(&cvHalves[0], &bufHalves[0], key, Counter, Flags)
	numChunks := uint64(buflen / chunkSize)
	if numChunks > 8 {
		compressChunksAVX2(&cvHalves[1], &bufHalves[1], key, Counter+8, Flags)
	}
	if buflen%chunkSize != 0 {
		// use non-asm for remainder
		partialChunk := buf[buflen-buflen%chunkSize : buflen]
		cvs[numChunks] = chainingValue(compressChunk(partialChunk, key, Counter+numChunks, Flags))
		numChunks++
	}
	return mergeSubtrees(&cvs, numChunks, key, Flags)
}

func compressBuffer(buf *[maxSIMD * chunkSize]byte, buflen int, key *[8]uint32, Counter uint64, Flags uint32) Node {
	switch {
	case haveAVX512 && buflen >= chunkSize*2:
		return compressBufferAVX512(buf, buflen, key, Counter, Flags)
	case haveAVX2 && buflen >= chunkSize*2:
		return compressBufferAVX2(buf, buflen, key, Counter, Flags)
	default:
		return compressBufferGeneric(buf, buflen, key, Counter, Flags)
	}
}

func compressChunk(chunk []byte, key *[8]uint32, Counter uint64, Flags uint32) Node {
	n := Node{
		CV:       *key,
		Counter:  Counter,
		BlockLen: blockSize,
		Flags:    Flags | flagChunkStart,
	}
	blockBytes := (*[64]byte)(unsafe.Pointer(&n.Block))[:]
	for len(chunk) > blockSize {
		copy(blockBytes, chunk)
		chunk = chunk[blockSize:]
		n.CV = chainingValue(n)
		n.Flags &^= flagChunkStart
	}
	// pad last Block with zeros
	n.Block = [16]uint32{}
	copy(blockBytes, chunk)
	n.BlockLen = uint32(len(chunk))
	n.Flags |= flagChunkEnd
	return n
}

func hashBlock(out *[64]byte, buf []byte) {
	var Block [16]uint32
	copy((*[64]byte)(unsafe.Pointer(&Block))[:], buf)
	CompressNodeGeneric((*[16]uint32)(unsafe.Pointer(out)), Node{
		CV:       iv,
		Block:    Block,
		BlockLen: uint32(len(buf)),
		Flags:    flagChunkStart | flagChunkEnd | flagRoot,
	})
}

func compressBlocks(out *[maxSIMD * blockSize]byte, n Node) {
	switch {
	case haveAVX512:
		compressBlocksAVX512(out, &n.Block, &n.CV, n.Counter, n.BlockLen, n.Flags)
	case haveAVX2:
		outs := (*[2][512]byte)(unsafe.Pointer(out))
		compressBlocksAVX2(&outs[0], &n.Block, &n.CV, n.Counter, n.BlockLen, n.Flags)
		compressBlocksAVX2(&outs[1], &n.Block, &n.CV, n.Counter+8, n.BlockLen, n.Flags)
	default:
		outs := (*[maxSIMD][64]byte)(unsafe.Pointer(out))
		compressBlocksGeneric(outs, n)
	}
}

func mergeSubtrees(cvs *[maxSIMD][8]uint32, numCVs uint64, key *[8]uint32, Flags uint32) Node {
	if !haveAVX2 {
		return mergeSubtreesGeneric(cvs, numCVs, key, Flags)
	}
	for numCVs > 2 {
		if numCVs%2 == 0 {
			compressParentsAVX2((*[8][8]uint32)(unsafe.Pointer(cvs)), cvs, key, Flags)
		} else {
			keep := cvs[numCVs-1]
			compressParentsAVX2((*[8][8]uint32)(unsafe.Pointer(cvs)), cvs, key, Flags)
			cvs[numCVs/2] = keep
			numCVs++
		}
		numCVs /= 2
	}
	return parentNode(cvs[0], cvs[1], *key, Flags)
}

func wordsToBytes(words [16]uint32, Block *[64]byte) {
	*Block = *(*[64]byte)(unsafe.Pointer(&words))
}
