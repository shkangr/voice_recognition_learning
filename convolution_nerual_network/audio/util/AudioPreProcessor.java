package org.deeplearning4j.examples.convolution.mnist.audio.util;

import java.io.IOException;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;


public class AudioPreProcessor
{
  public static final float DEFAULT_SAMPLE_RATE = 11025.0f;

  //information about the streams
  protected AudioInputStream in;
  protected float sampleRate;
  protected int frameSize;
  protected boolean isBigEndian;

  //implementation fields
  private byte[] inputBuffer = new byte[1024];
  private double normalise;
  private double scale;
  private double dB_max;

  public AudioPreProcessor(AudioInputStream in, float sampleRate) throws IllegalArgumentException
  {
    AudioFormat source, converted;

    //check for not null
    if(in == null)
      throw new IllegalArgumentException("the inputstream must not be a  null value");

    //check sample rate
    source = in.getFormat();
    sampleRate = sampleRate;
    if(sampleRate < 0 || sampleRate > source.getSampleRate())
      throw new IllegalArgumentException("the sample rate to convert to must be greater null and less or equal the sample rate of the input stream");

    //convert input stream to pcm
    this.in = convertToPCM(in);

    try
    {
      //first try to use a optimized version
      this.in = new ReducedAudioInputStream(this.in, sampleRate);
    }
    catch(IllegalArgumentException iae)
    {
      //convert sample rate (by using the java sound api)
      this.in = convertSampleRate(this.in, sampleRate);

      //reduce number of channels
      this.in = convertChannels(this.in, 1);
    }

    //get information about the stream
    converted = this.in.getFormat();

    //check conversion
    if(converted.getSampleSizeInBits() != 8 && converted.getSampleSizeInBits() != 16 && converted.getSampleSizeInBits() != 24 && converted.getSampleSizeInBits() != 32)
      throw new IllegalArgumentException("the sample size of the input stream must be 8, 16, 24 or 32 bit");
    if(converted.getFrameSize() != (converted.getSampleSizeInBits()/8) || converted.getSampleRate() != sampleRate)
      throw new IllegalArgumentException("the conversion is not supported");

    //set fields
    this.sampleRate = sampleRate;
    this.frameSize = converted.getFrameSize();
    this.isBigEndian = converted.isBigEndian();

    //set dB_max
    dB_max = 6.0d * converted.getSampleSizeInBits();

    //bits - 1 because positive and negatve values are possible
    this.normalise = (double) converted.getChannels() * (1 << (converted.getSampleSizeInBits() - 1));

    //compute rescale factor to rescale and normalise at once (default is 96dB = 2^16)
    this.scale = (Math.pow(10, dB_max / 20)) / normalise;
  }

  public AudioPreProcessor(AudioInputStream in) throws IllegalArgumentException
  {
      this(in, DEFAULT_SAMPLE_RATE);
  }

  public float getSampleRate()
  {
    return sampleRate;
  }

  public int skip(int len) throws IllegalArgumentException, IOException
  {
    int bytesRead = 0;
    int bytesToRead = 0;
    int total = 0;

    //check len
    if(len < 0)
      throw new IllegalArgumentException("len must be a positve value");

    //skip by reading blocks of the size of the buffer
    int blockSize = inputBuffer.length;

    //compute byte to read
    bytesToRead = len * frameSize;

    //read new data from inputstream
    while(total < bytesToRead)
    {
      //adjust blockSize for the last block to skip
      if(blockSize > bytesToRead - total)
        blockSize = bytesToRead - total;

      //read one block of data
      bytesRead = read(blockSize);

      //compute total bytes read
      total += bytesRead;

      //check if we read the correct number of bytes
      if(blockSize != bytesRead)
        return total/frameSize;
    }

    //return number of samples read
    return total/frameSize;
  }

  public int fastSkip(int len) throws IllegalArgumentException, IOException
  {
    if(len < 0)
      throw new IllegalArgumentException("the number of frames to skip must be positiv");

    if(len == 0)
      return 0;

    return ((int)in.skip(len*frameSize))/frameSize;
  }

  public double[] get(int len) throws IllegalArgumentException, IOException
  {
    int bytesRead = 0;
    int bytesToRead = 0;
    double[] outputData;

    //check len
    if(len < 0)
      throw new IllegalArgumentException("len must be a positive value");

    //compute byte to read
    bytesToRead = len * frameSize;

    //read new data from inputstream
    bytesRead = read(bytesToRead);

    //allocate array for data
    outputData = new double[bytesRead/frameSize];

    //now calculate double values
    convertToDouble(inputBuffer, bytesRead, outputData, 0);

    return outputData;
  }

  public int append(double[] buffer, int start, int len) throws IllegalArgumentException, IOException
  {
    int bytesRead = 0;
    int bytesToRead = 0;

    //check start and len
    if(start < 0 || len < 0)
      throw new IllegalArgumentException("start and len must be positiv values");

    //check the buffer size
    if(buffer == null || buffer.length - start < len)
      throw new IllegalArgumentException("Specified buffer is too samll to hold all samples.");

    //compute byte to read
    bytesToRead = len * frameSize;

    //read new data from inputstream
    bytesRead = read(bytesToRead);

    //now append new data
    convertToDouble(inputBuffer, bytesRead, buffer, start);

    return bytesRead/frameSize;
  }

  protected void convertToDouble(byte[] in, int len, double[] out, int start)
  {
    int db = 0;

    if(frameSize == 1)
    {
      //8-bit signed PCM
      for (int i = 0; i < len; i += frameSize)
        out[start++] = ((double) in[i]) * scale;
    }
    else
    {
      //more than one byte per sample
      if(isBigEndian)
      {
        //the bytes of one sample value are in big-Endian order
        //first byte will be converted to int with respect to the sign
        db = (int) in[0];
        for (int i = 1, j = frameSize; i < len; i++)
        {
          if(i == j)//finished one sample?
          {
            //convert to double value (and rescale for compatibility to the matlab implementation)
            out[start++] = ((double) db) * scale;
            //first byte will be converted to int with respect to the sign
            db = (int) in[i];
            //set the end of the next sample value
            j += frameSize;
          }
          else
          {
            //combine the bytes of the sample (unsigned bytes)
            db = db << 8 | ((int) in[i] & 0xff);
          }
        }
        //convert to double value (and rescale for compatibility to the matlab implementation)
        out[start++] = ((double) db) * scale;
      }
      else
      {
        //the bytes of one sample value are in little-Endian order
        for (int i = 0; i < len; i += frameSize)
        {
          //first byte will be converted to int with respect to the sign
          db = (int) in[i + frameSize - 1];

          //combine the bytes of the sample (unsigned bytes) in reversed order
          for (int b = frameSize - 2; b >= 0; b--)
            db = db << 8 | ((int) in[i + b] & 0xff);

          //convert to double value (and rescale for compatibility to the matlab implementation)
          out[start++] = ((double) db) * scale;
        }
      }
    }
  }

  static public AudioInputStream convertToPCM(AudioInputStream in) throws IllegalArgumentException
  {
    AudioFormat targetFormat = null;
    AudioInputStream pcm = in;
    AudioFormat sourceFormat = in.getFormat();
    int sampleSizeInBits;

    //check for not null
    if(in == null)
      throw new IllegalArgumentException("the inputstream must not be null values");

    //set appropriate sample size
    sampleSizeInBits = sourceFormat.getSampleSizeInBits();
    if(sampleSizeInBits == -1)
      sampleSizeInBits = 16;
    if(sampleSizeInBits != 8 && sampleSizeInBits != 16 && sampleSizeInBits != 24 && sampleSizeInBits != 32)
      sampleSizeInBits = 16;

    //get source format
    sourceFormat = in.getFormat();

    //convert to pcm
    if (sourceFormat.getEncoding() != AudioFormat.Encoding.PCM_SIGNED)
    {


      targetFormat = new AudioFormat(AudioFormat.Encoding.PCM_SIGNED,
                                         sourceFormat.getSampleRate(),
                                         sampleSizeInBits,
                                         sourceFormat.getChannels(),
                                         sourceFormat.getChannels() * (sampleSizeInBits/8),
                                         sourceFormat.getSampleRate(),
                                         false);

        //convert to pcm
        if (AudioSystem.isConversionSupported(targetFormat, sourceFormat))
          pcm = AudioSystem.getAudioInputStream(targetFormat, pcm);
        else
          throw new IllegalArgumentException("conversion to PCM_SIGNED not supported for this input stream");
    }
    return pcm;
  }

  public static AudioInputStream convertSampleRate(AudioInputStream in, float sampleRate) throws IllegalArgumentException
  {
      AudioInputStream converted;
      AudioFormat sourceFormat, targetFormat;

      //check for not null
      if(in == null)
        throw new IllegalArgumentException("the inputstream must not be null values");

      //check sample rate
      if(sampleRate < 0 || sampleRate > in.getFormat().getSampleRate())
        throw new IllegalArgumentException("the sample rate to convert to must be greater null and less or equal the sample rate of the input stream");

      converted = in;

      sourceFormat = in.getFormat();
      targetFormat = new AudioFormat(AudioFormat.Encoding.PCM_SIGNED,
                                   sampleRate,
                                   sourceFormat.getSampleSizeInBits(),
                                   sourceFormat.getChannels(),
                                   sourceFormat.getFrameSize(),
                                   sampleRate,
                                   false);

      if (sourceFormat.getSampleRate() != targetFormat.getSampleRate())
      {
        if (AudioSystem.isConversionSupported(targetFormat, sourceFormat))
          converted = AudioSystem.getAudioInputStream(targetFormat, in);
        else
          throw new IllegalArgumentException("Conversion to specified sample rate not supported.");
      }
      return converted;
  }

  public static AudioInputStream convertChannels(AudioInputStream in, int channels) throws IllegalArgumentException
  {
    AudioInputStream converted;
    AudioFormat sourceFormat, targetFormat;

    //check for not null
    if(in == null)
      throw new IllegalArgumentException("the inputstream must not be null values");

    //check number of channels
    if(channels < 1)
      throw new IllegalArgumentException("the number of channels must be greater than one");

    converted = in;

    //convert channels
    sourceFormat = in.getFormat();
    targetFormat = new AudioFormat(sourceFormat.getSampleRate(),
                                   sourceFormat.getSampleSizeInBits(),
                                   channels,
                                   true,
                                   sourceFormat.isBigEndian());

    if (sourceFormat.getChannels() != targetFormat.getChannels())
    {
      if (AudioSystem.isConversionSupported(targetFormat, sourceFormat))
        converted = AudioSystem.getAudioInputStream(targetFormat, converted);
      else
        throw new IllegalArgumentException("Conversion to specified number of channels not supported.");
    }

    return converted;
 }

 public static AudioInputStream convertBitsPerSample(AudioInputStream in, int bitsPerSample) throws IllegalArgumentException
 {
   AudioInputStream converted;
   AudioFormat sourceFormat, targetFormat;

   //check for not null
   if(in == null)
      throw new IllegalArgumentException("the inputstream must not be null values");

  //check number of bits per sample
  if(bitsPerSample!=8 && bitsPerSample!=16 && bitsPerSample!=24 && bitsPerSample!=32)
    throw new IllegalArgumentException("number of bits must be 8, 16, 24 or 32");

   converted = in;

   sourceFormat = in.getFormat();
   targetFormat = new AudioFormat(sourceFormat.getSampleRate(), bitsPerSample,
                               sourceFormat.getChannels(), true,
                               sourceFormat.isBigEndian());

   if (targetFormat.getSampleSizeInBits() != sourceFormat.getSampleSizeInBits())
   {
     if (AudioSystem.isConversionSupported(targetFormat, sourceFormat))
       converted = AudioSystem.getAudioInputStream(targetFormat, converted);
     else
       throw new IllegalArgumentException("conversion to specified sample size (bits) not supported");
   }

   return converted;
 }

  public static AudioInputStream convertByteOrder(AudioInputStream in, boolean isBigEndian) throws IllegalArgumentException
  {
    AudioInputStream converted;
    AudioFormat sourceFormat, targetFormat;

    //check for not null
    if(in == null)
      throw new IllegalArgumentException("the inputstream must not be null values");

    converted = in;

    sourceFormat = in.getFormat();
    targetFormat = new AudioFormat(sourceFormat.getSampleRate() , sourceFormat.getSampleSizeInBits(), sourceFormat.getChannels(), true, isBigEndian);
    if(targetFormat.isBigEndian() != sourceFormat.isBigEndian())
    {
      if (AudioSystem.isConversionSupported(targetFormat, sourceFormat))
        converted = AudioSystem.getAudioInputStream(targetFormat, converted);
      else
        throw new IllegalArgumentException("conversion of byte order not supported");
    }

    return converted;
  }

  private int read(int bytesToRead) throws
    IOException
  {
      int nBytesRead = 0;
      int nBytesTotalRead = 0;

      //check buffer size and adjust it
      if(inputBuffer.length < bytesToRead)
        inputBuffer = new byte[bytesToRead];

      while (nBytesRead != -1 && bytesToRead > 0)
      {
        nBytesRead = in.read(inputBuffer, nBytesTotalRead, bytesToRead);
        if (nBytesRead != -1)
        {
          bytesToRead -= nBytesRead;
          nBytesTotalRead += nBytesRead;
        }
      }

      return nBytesTotalRead;
  }
}
