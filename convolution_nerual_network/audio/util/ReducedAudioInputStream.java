package org.deeplearning4j.examples.convolution.mnist.audio.util;

import java.io.IOException;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;

public class ReducedAudioInputStream extends AudioInputStream
{
  //fields
  protected AudioInputStream in;
  protected AudioFormat sourceFormat;
  protected AudioFormat targetFormat;
  protected int frameSize = 0;
  protected int sampleSize = 0;

  //implementation fields
  private int sampleShift = 0;
  private int pos = 0;
  private int skipSize = 0;
  private int ratio = 0;
  private int offsetHigh = 0;
  private int offsetLow = 0;
  private int bytesInBuffer = 0;
  private boolean endOfFile = false;
  private boolean firstRead = true;
  private boolean reduceChannels = false;
  private byte[] inputBuffer;

  //protected final int INITIAL_BUFFER_SIZE = 196608;
  protected final int INITIAL_BUFFER_SIZE = 10240;


  public ReducedAudioInputStream(AudioInputStream ais, float targetSampleRate)
  {
    super(ais, ais.getFormat(), ais.getFrameLength());

    //set audio input stream and audio format
    this.in = ais;
    this.sourceFormat = ais.getFormat();

    //check the number of channels
    if(sourceFormat.getChannels() != 1 && sourceFormat.getChannels() != 2)
      throw new IllegalArgumentException("audio input stream must have one or two channels");

    //set flag indicating, if we have to reduce the number of channels
    if(sourceFormat.getChannels() ==2)
      reduceChannels = true;
    else
      reduceChannels = false;

    //check if the input stream is a pcm stream
    if(sourceFormat.getEncoding() != AudioFormat.Encoding.PCM_SIGNED)
      throw new IllegalArgumentException("encoding of audio input stream must be PCM_SIGNED");

    //check if the number of bits is valid
    if(sourceFormat.getSampleSizeInBits() != 16 && sourceFormat.getSampleSizeInBits() != 24 && sourceFormat.getSampleSizeInBits() != 32)
      throw new IllegalArgumentException("sample size in bits of audio input stream must be 16, 24 or 32");

    //frame and sample rate must be the same
    if(sourceFormat.getFrameRate() != sourceFormat.getSampleRate())
      throw new IllegalArgumentException("frame rate and sample rate are not equal");

    //compute internal fields
    ratio = (int) Math.floor(sourceFormat.getSampleRate()/targetSampleRate);
    frameSize = ratio * sourceFormat.getFrameSize();
    sampleSize = sourceFormat.getSampleSizeInBits()/8;
    skipSize = frameSize - (sampleSize * sourceFormat.getChannels());
    sampleShift = sourceFormat.getSampleSizeInBits() - 16;

    //check if the conversion ratio is an integer value
    if(ratio != format.getFrameRate()/targetSampleRate)
      throw new IllegalArgumentException("the sample rate conversion ratio muste be a integer value");

    //create the target format, that this input stream will return
    targetFormat = new AudioFormat(sourceFormat.getEncoding(),
                                   targetSampleRate,
                                   sourceFormat.getSampleSizeInBits(),
                                   1,
                                   sourceFormat.getSampleSizeInBits()/8,
                                   targetSampleRate,
                                   false);

    //check if the conversion is usefull
    if(ratio == 1 && sampleShift == 0 && sourceFormat.getChannels() == 2)
      throw new IllegalArgumentException("using this wrapper is useless, since audio input stream is allready reduced");

    //create buffer
    inputBuffer = new byte[INITIAL_BUFFER_SIZE * frameSize];

    //take care of big/little endian
    if(sourceFormat.isBigEndian())
    {
      //big endian
      offsetHigh = 0;
      offsetLow = 1;
    }
    else
    {
      //little endian
      offsetHigh = sampleSize - 1;
      offsetLow = sampleSize - 2;
    }
  }


  public int available() throws IOException
  {
    if(reduceChannels)
      return (super.available()/ratio)/2;
    else
      return super.available()/ratio;
  }


  public void close() throws IOException
  {
    super.close();
  }

  public boolean markSupported()
  {
    return false;
  }

  public int read() throws IOException
  {
    throw new IllegalArgumentException("this method is not yet supported");
  }

  public int read(byte[] b) throws IOException
  {
    //special action if this is the first read operation
    if(firstRead)
    {
      firstRead = false;

      //fill the buffer
      refillBuffer();

      //take the last sample of the first frame
      pos = skipSize;

      //add two bytes empty to the stream (for compatibility with Tritonus)
      if(ratio != 1)
      {
        b[0] = 0;
        b[1] = 0;

        return readBytesFormBuffer(b, 2, b.length - 2) + 2;
      }
      else
      {
        return readBytesFormBuffer(b, 0, b.length);
      }
    }
    else
    {
      //just return
      return readBytesFormBuffer(b, 0, b.length);
    }
  }

  public int read(byte[] b, int off, int len) throws IOException
  {
    //special action if this is the first read operation
    if(firstRead)
    {
      firstRead = false;

      //fill the bufer
      refillBuffer();

      //take the last sample of the first frame
      pos = skipSize;

      //add two bytes empty to the stream (for compatibility with Tritonus)
      if(ratio != 1)
      {
        b[off++] = 0;
        b[off++] = 0;

        return readBytesFormBuffer(b, off, len - 2) + 2;
      }
      else
      {
        return readBytesFormBuffer(b, off, len);
      }
    }
    else
    {
      //just return
      return readBytesFormBuffer(b, off, len);
    }
  }

  public long skip(long n) throws IOException
  {
    long totalSkipped = 0;
    long skipped;

    // loop until bytes are really skipped.
    while (totalSkipped < n)
    {
      skipped = in.skip(n-totalSkipped);

      if (totalSkipped ==-1)
        throw new IOException("skip not supported");
      else if(skipped == 0)
        return 0;
      else
        totalSkipped += totalSkipped + skipped;
    }

    //change the firstRead flag
    if(firstRead)
      firstRead = false;

    return totalSkipped;
  }

  public AudioFormat getFormat()
  {
    return targetFormat;
  }

  public long getFrameLength()
  {
    if(in.getFrameLength() == -1)
      return -1;
    else
      return super.getFrameLength()/ratio;
  }

  private int read(int bytesToRead, int offset) throws
    IOException
  {
      int nBytesRead = 0;
      int nBytesTotalRead = 0;

      //check buffer size and adjust it
      if(inputBuffer.length < bytesToRead + offset)
        inputBuffer = new byte[bytesToRead + offset];

      while (nBytesRead != -1 && bytesToRead > 0)
      {
        nBytesRead = in.read(inputBuffer, offset + nBytesTotalRead, bytesToRead);
        if (nBytesRead != -1)
        {
          bytesToRead -= nBytesRead;
          nBytesTotalRead += nBytesRead;
        }
      }

      if(nBytesRead == -1)
        endOfFile = true;

      return nBytesTotalRead;
  }

  private int readBytesFormBuffer(byte[] buffer, int start, int bytesToRead) throws IOException
  {
      int sample = 0;
      int bytesRead = 0;

      while(bytesRead < bytesToRead)
      {
        //read left channel
        sample = readSample();

        if(reduceChannels)
        {
          //there are two channels and therefore two samples per frame
          sample += readSample();

          //prevent overflow
          if(sample > 32767)
            sample = 32767;
          if(sample < -32768)
            sample = -32768;
        }

        //check if we still read vaild data
        if(pos < bytesInBuffer)
        {
          //write sample to output buffer
          buffer[start++] = (byte) (sample & 0xff);
          buffer[start++] = (byte) ((sample >> 8) & 0xff);
          bytesRead += 2;

          //jump to next frame
          pos += skipSize;
        }
        else
        {
          //undo last readSample operation
          if(reduceChannels)
            pos -= 2*sampleSize;
          else
            pos -= sampleSize;

          //try to fill buffer
          if (endOfFile)
          {
            if(bytesRead > 0)
              return bytesRead;
            else
              return -1;
          }
          else
          {
            refillBuffer();
          }
        }
      }

      return bytesRead;
  }

  private int readSample()
  {
    //high byte will be converted to int with respect to the sign
    int sample = (int) inputBuffer[pos + offsetHigh];
    //combine with low byte, ignore others (16 bit)
    sample = (sample << 8) | ((int) inputBuffer[pos + offsetLow] & 0xff);

    //ajust position
    pos += sampleSize;

    return sample;
  }

  private void refillBuffer() throws IOException
  {
    int nBytesToRead, nBytesRead;

    //copy last few bytes to the beginning of the array
    for(nBytesToRead = 0; pos < bytesInBuffer; nBytesToRead++, pos++)
      inputBuffer[nBytesToRead] = inputBuffer[pos];

    //save number of bytes in buffer
    pos = nBytesToRead;

    //compute the number of bytes to read
    nBytesToRead = (inputBuffer.length - nBytesToRead) - frameSize;

    //fill buffer
    nBytesRead = read(nBytesToRead, pos);

    //ajust buffer size and buffer position
    bytesInBuffer = pos + nBytesRead;
    pos = 0;
  }
}
