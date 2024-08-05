import { NextRequest, NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import { PDFDocument } from 'pdf-lib';
import Tesseract from 'tesseract.js';
import { getEmbeddingsTransformer, searchArgs } from '@/utils/openai';
import { MongoDBAtlasVectorSearch } from '@langchain/community/vectorstores/mongodb_atlas';
import { CharacterTextSplitter } from 'langchain/text_splitter';

export async function POST(req: NextRequest) {
  try {
    const formData: FormData = await req.formData();
    const uploadedFiles = formData.getAll('filepond');
    let fileName = '';
    let parsedText = '';

    if (uploadedFiles && uploadedFiles.length > 0) {
      const uploadedFile = uploadedFiles[1];
      console.log('Uploaded file:', uploadedFile);

      if (uploadedFile instanceof File) {
        fileName = uploadedFile.name.toLowerCase();
        const tempFilePath = `/tmp/${fileName}.pdf`;
        const fileBuffer = Buffer.from(await uploadedFile.arrayBuffer());
        await fs.writeFile(tempFilePath, fileBuffer);
        let dataBuffer = await fs.readFile(tempFilePath);

        const pdfDoc = await PDFDocument.load(dataBuffer);
        const pages = pdfDoc.getPages();

        for (const page of pages) {
          const { width, height } = page.getSize();
          const pdfImage = await page.embedPng(page.render({ width, height }).toBuffer());
          const imageBuffer = pdfImage.toBuffer();

          // Use Tesseract to recognize text from image
          const { data: { text } } = await Tesseract.recognize(imageBuffer, 'eng');
          parsedText += text;
        }

        const chunks = await new CharacterTextSplitter({
          separator: "\n",
          chunkSize: 1000,
          chunkOverlap: 100
        }).splitText(parsedText);
        console.log(chunks.length);

        await MongoDBAtlasVectorSearch.fromTexts(
          chunks, [],
          getEmbeddingsTransformer(),
          searchArgs()
        );
        
        return NextResponse.json({ message: "Uploaded to MongoDB" }, { status: 200 });
      } else {
        console.log('Uploaded file is not in the expected format.');
        return NextResponse.json({ message: 'Uploaded file is not in the expected format' }, { status: 500 });
      }
    } else {
      console.log('No files found.');
      return NextResponse.json({ message: 'No files found' }, { status: 500 });
    }
  } catch (error) {
    console.error('Error processing request:', error);
    return new NextResponse("An error occurred during processing.", { status: 500 });
  }
}
