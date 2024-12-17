/***
 * @Author: xingwg
 * @Date: 2024-11-01 15:56:48
 * @LastEditTime: 2024-11-13 11:17:49
 * @FilePath: /dmnn2/include/codecs/nv_decoder.h
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#pragma once
#include "error_check.h"
#include "logging.h"
#include "nvcuvid.h"
#include <cuda.h>
#include <sstream>

#define MAX_FRM_CNT 32

namespace nv {
struct Rect {
    int l{0}, t{0}, r{0}, b{0};
};
struct Dim {
    int w{0};
    int h{0};
};
class NvDecoder {
public:
    /**
     *  @brief This function is used to initialize the decoder session.
     *  Application must call this function to initialize the decoder, before
     * starting to decode any frames.
     */
    NvDecoder(CUcontext cuContext, bool bUseDeviceFrame, cudaVideoCodec eCodec,
              bool bLowLatency = false, bool bDeviceFramePitched = false,
              const Rect *pCropRect = nullptr, const Dim *pResizeDim = nullptr,
              bool extract_user_SEI_Message = false, int maxWidth = 0,
              int maxHeight = 0, unsigned int clkRate = 1000,
              bool force_zero_latency = false);
    ~NvDecoder();

    /**
     *  @brief  This function is used to get the current CUDA context.
     */
    CUcontext GetContext() { return m_cuContext; }

    /**
     *  @brief  This function is used to get the output frame width.
     *  NV12/P016 output format width is 2 byte aligned because of U and V
     * interleave
     */
    int GetWidth() {
        LOG_ASSERT(m_nWidth);
        return (m_eOutputFormat == cudaVideoSurfaceFormat_NV12 ||
                m_eOutputFormat == cudaVideoSurfaceFormat_P016)
                   ? (m_nWidth + 1) & ~1
                   : m_nWidth;
    }

    /**
     *  @brief  This function is used to get the actual decode width
     */
    int GetDecodeWidth() {
        LOG_ASSERT(m_nWidth);
        return m_nWidth;
    }

    /**
     *  @brief  This function is used to get the output frame height (Luma
     * height).
     */
    int GetHeight() {
        LOG_ASSERT(m_nLumaHeight);
        return m_nLumaHeight;
    }

    /**
     *  @brief  This function is used to get the current chroma height.
     */
    int GetChromaHeight() {
        LOG_ASSERT(m_nChromaHeight);
        return m_nChromaHeight;
    }

    /**
     *  @brief  This function is used to get the number of chroma planes.
     */
    int GetNumChromaPlanes() {
        LOG_ASSERT(m_nNumChromaPlanes);
        return m_nNumChromaPlanes;
    }

    /**
     *   @brief  This function is used to get the current frame size based on
     * pixel format.
     */
    int GetFrameSize() {
        LOG_ASSERT(m_nWidth);
        return GetWidth() *
               (m_nLumaHeight + (m_nChromaHeight * m_nNumChromaPlanes)) *
               m_nBPP;
    }

    /**
     *   @brief  This function is used to get the current frame Luma plane size.
     */
    int GetLumaPlaneSize() {
        LOG_ASSERT(m_nWidth);
        return GetWidth() * m_nLumaHeight * m_nBPP;
    }

    /**
     *   @brief  This function is used to get the current frame chroma plane
     * size.
     */
    int GetChromaPlaneSize() {
        LOG_ASSERT(m_nWidth);
        return GetWidth() * (m_nChromaHeight * m_nNumChromaPlanes) * m_nBPP;
    }

    /**
     *  @brief  This function is used to get the pitch of the device buffer
     * holding the decoded frame.
     */
    int GetDeviceFramePitch() {
        LOG_ASSERT(m_nWidth);
        return m_nDeviceFramePitch ? (int)m_nDeviceFramePitch
                                   : GetWidth() * m_nBPP;
    }

    /**
     *   @brief  This function is used to get the bit depth associated with the
     * pixel format.
     */
    int GetBitDepth() {
        LOG_ASSERT(m_nWidth);
        return m_nBitDepthMinus8 + 8;
    }

    /**
     *   @brief  This function is used to get the bytes used per pixel.
     */
    int GetBPP() {
        LOG_ASSERT(m_nWidth);
        return m_nBPP;
    }

    /**
     *   @brief  This function is used to get the YUV chroma format
     */
    cudaVideoSurfaceFormat GetOutputFormat() { return m_eOutputFormat; }

    /**
     *   @brief  This function is used to get information about the video stream
     * (codec, display parameters etc)
     */
    CUVIDEOFORMAT GetVideoFormatInfo() {
        LOG_ASSERT(m_nWidth);
        return m_videoFormat;
    }

    /**
     *   @brief  This function is used to get codec string from codec id
     */
    const char *GetCodecString(cudaVideoCodec eCodec);

    /**
     *   @brief  This function is used to print information about the video
     * stream
     */
    std::string GetVideoInfo() const { return m_videoInfo.str(); }

    /**
     *   @brief  This function decodes a frame and returns the number of frames
     * that are available for display. All frames that are available for display
     * should be read before making a subsequent decode call.
     *   @param  pData - pointer to the data buffer that is to be decoded
     *   @param  nSize - size of the data buffer in bytes
     *   @param  nFlags - CUvideopacketflags for setting decode options
     *   @param  nTimestamp - presentation timestamp
     */
    int Decode(const uint8_t *pData, int nSize, int nFlags = 0,
               int64_t nTimestamp = 0);

    /**
     *   @brief  This function returns a decoded frame and timestamp. This
     * function should be called in a loop for fetching all the frames that are
     * available for display.
     */
    uint8_t *GetFrame(int64_t *pTimestamp = nullptr);

    /**
     *   @brief  This function decodes a frame and returns the locked frame
     * buffers This makes the buffers available for use by the application
     * without the buffers getting overwritten, even if subsequent decode calls
     * are made. The frame buffers remain locked, until UnlockFrame() is called
     */
    uint8_t *GetLockedFrame(int64_t *pTimestamp = nullptr);

    /**
     *   @brief  This function unlocks the frame buffer and makes the frame
     * buffers available for write again
     *   @param  ppFrame - pointer to array of frames that are to be unlocked
     *   @param  nFrame - number of frames to be unlocked
     */
    void UnlockFrame(uint8_t **pFrame);

    /**
     *   @brief  This function allows app to set decoder reconfig params
     *   @param  pCropRect - cropping rectangle coordinates
     *   @param  pResizeDim - width and height of resized output
     */
    int setReconfigParams(const Rect *pCropRect, const Dim *pResizeDim);

    /**
     *   @brief  This function allows app to set operating point for AV1 SVC
     * clips
     *   @param  opPoint - operating point of an AV1 scalable bitstream
     *   @param  bDispAllLayers - Output all decoded frames of an AV1 scalable
     * bitstream
     */
    void SetOperatingPoint(const uint32_t opPoint, const bool bDispAllLayers) {
        m_nOperatingPoint = opPoint;
        m_bDispAllLayers = bDispAllLayers;
    }

    // start a timer
    // void startTimer() { m_stDecode_time.Start(); }

    // stop the timer
    // double stopTimer() { return m_stDecode_time.Stop(); }

    void setDecoderSessionID(int sessionID) { decoderSessionID = sessionID; }
    int getDecoderSessionID() { return decoderSessionID; }

    // Session overhead refers to decoder initialization and deinitialization
    // time
    static void addDecoderSessionOverHead(int sessionID, int64_t duration) {
        sessionOverHead[sessionID] += duration;
    }
    static int64_t getDecoderSessionOverHead(int sessionID) {
        return sessionOverHead[sessionID];
    }

private:
    int decoderSessionID;  // Decoder session identifier. Used to gather session
                           // level stats.
    static std::map<int, int64_t>
        sessionOverHead;  // Records session overhead of
                          // initialization+deinitialization time. Format is
                          // (thread id, duration)
    /**
     * @brief  Callback function to be registered for getting a callback when
     * decoding of sequence starts
     */
    static int CUDAAPI HandleVideoSequenceProc(void *pUserData,
                                               CUVIDEOFORMAT *pVideoFormat) {
        return ((NvDecoder *)pUserData)->HandleVideoSequence(pVideoFormat);
    }

    /**
     * @brief  Callback function to be registered for getting a callback when a
     * decoded frame is ready to be decoded
     */
    static int CUDAAPI HandlePictureDecodeProc(void *pUserData,
                                               CUVIDPICPARAMS *pPicParams) {
        return ((NvDecoder *)pUserData)->HandlePictureDecode(pPicParams);
    }

    /**
     * @brief  Callback function to be registered for getting a callback when a
     * decoded frame is available for display
     */
    static int CUDAAPI
    HandlePictureDisplayProc(void *pUserData, CUVIDPARSERDISPINFO *pDispInfo) {
        return ((NvDecoder *)pUserData)->HandlePictureDisplay(pDispInfo);
    }

    /**
     * @brief  Callback function to be registered for getting a callback to get
     * operating point when AV1 SVC sequence header start.
     */
    static int CUDAAPI HandleOperatingPointProc(
        void *pUserData, CUVIDOPERATINGPOINTINFO *pOPInfo) {
        return ((NvDecoder *)pUserData)->GetOperatingPoint(pOPInfo);
    }

    /**
     * @brief  Callback function to be registered for getting a callback when
     * all the unregistered user SEI Messages are parsed for a frame.
     */
    static int CUDAAPI HandleSEIMessagesProc(
        void *pUserData, CUVIDSEIMESSAGEINFO *pSEIMessageInfo) {
        return ((NvDecoder *)pUserData)->GetSEIMessage(pSEIMessageInfo);
    }

    /**
     * @brief  This function gets called when a sequence is ready to be decoded.
     * The function also gets called when there is format change
     */
    int HandleVideoSequence(CUVIDEOFORMAT *pVideoFormat);

    /**
     * @brief  This function gets called when a picture is ready to be decoded.
     * cuvidDecodePicture is called from this function to decode the picture
     */
    int HandlePictureDecode(CUVIDPICPARAMS *pPicParams);

    /**
     * @brief  This function gets called after a picture is decoded and
     * available for display. Frames are fetched and stored in internal buffer
     */
    int HandlePictureDisplay(CUVIDPARSERDISPINFO *pDispInfo);

    /**
     * @brief  This function gets called when AV1 sequence encounter more than
     * one operating points
     */
    int GetOperatingPoint(CUVIDOPERATINGPOINTINFO *pOPInfo);

    /**
     * @brief  This function gets called when all unregistered user SEI messages
     * are parsed for a frame
     */
    int GetSEIMessage(CUVIDSEIMESSAGEINFO *pSEIMessageInfo);

    /**
     * @brief  This function reconfigure decoder if there is a change in
     * sequence params.
     */
    int ReconfigureDecoder(CUVIDEOFORMAT *pVideoFormat);

private:
    CUcontext m_cuContext = nullptr;
    CUvideoctxlock m_ctxLock;
    CUvideoparser m_hParser = nullptr;
    CUvideodecoder m_hDecoder = nullptr;
    bool m_bUseDeviceFrame;
    // dimension of the output
    unsigned int m_nWidth = 0, m_nLumaHeight = 0, m_nChromaHeight = 0;
    unsigned int m_nNumChromaPlanes = 0;
    // height of the mapped surface
    int m_nSurfaceHeight = 0;
    int m_nSurfaceWidth = 0;
    cudaVideoCodec m_eCodec = cudaVideoCodec_NumCodecs;
    cudaVideoChromaFormat m_eChromaFormat = cudaVideoChromaFormat_420;
    cudaVideoSurfaceFormat m_eOutputFormat = cudaVideoSurfaceFormat_NV12;
    int m_nBitDepthMinus8 = 0;
    int m_nBPP = 1;
    CUVIDEOFORMAT m_videoFormat = {};
    Rect m_displayRect = {};
    // stock of frames
    std::vector<uint8_t *> m_vpFrame;
    // timestamps of decoded frames
    std::vector<int64_t> m_vTimestamp;
    int m_nDecodedFrame = 0, m_nDecodedFrameReturned = 0;
    int m_nDecodePicCnt = 0, m_nPicNumInDecodeOrder[MAX_FRM_CNT];
    CUVIDSEIMESSAGEINFO *m_pCurrSEIMessage = nullptr;
    CUVIDSEIMESSAGEINFO m_SEIMessagesDisplayOrder[MAX_FRM_CNT];
    FILE *m_fpSEI = nullptr;
    bool m_bEndDecodeDone = false;
    std::mutex m_mtxVPFrame;
    int m_nFrameAlloc = 0;
    CUstream m_cuvidStream = 0;
    bool m_bDeviceFramePitched = false;
    size_t m_nDeviceFramePitch = 0;
    Rect m_cropRect = {};
    Dim m_resizeDim = {};

    std::ostringstream m_videoInfo;
    unsigned int m_nMaxWidth = 0, m_nMaxHeight = 0;
    bool m_bReconfigExternal = false;
    bool m_bReconfigExtPPChange = false;
    // StopWatch m_stDecode_time;

    unsigned int m_nOperatingPoint = 0;
    bool m_bDispAllLayers = false;
    // In H.264, there is an inherent display latency for video contents
    // which do not have num_reorder_frames=0 in the VUI. This applies to
    // All-Intra and IPPP sequences as well. If the user wants zero display
    // latency for All-Intra and IPPP sequences, the below flag will enable
    // the display callback immediately after the decode callback.
    bool m_bForce_zero_latency = false;
    bool m_bExtractSEIMessage = false;
};
}  // namespace nv
