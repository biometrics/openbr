/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2012 The MITRE Corporation                                      *
 *                                                                           *
 * Licensed under the Apache License, Version 2.0 (the "License");           *
 * you may not use this file except in compliance with the License.          *
 * You may obtain a copy of the License at                                   *
 *                                                                           *
 *     http://www.apache.org/licenses/LICENSE-2.0                            *
 *                                                                           *
 * Unless required by applicable law or agreed to in writing, software       *
 * distributed under the License is distributed on an "AS IS" BASIS,         *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
 * See the License for the specific language governing permissions and       *
 * limitations under the License.                                            *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <openbr/plugins/openbr_internal.h>

namespace br
{

using namespace cv;

/*!
 * \brief DOCUMENT ME
 * \author Unknown \cite unknown
 */
class vbbGallery : public Gallery
{
    Q_OBJECT

    void init()
    {
        MatlabIO matio;
        QString filename = (Globals->path.isEmpty() ? "" : Globals->path + "/") + file.name;
        bool ok = matio.open(filename.toStdString(), "r");
        if (!ok) qFatal("Couldn't open the vbb file");

        vector<MatlabIOContainer> variables;
        variables = matio.read();
        matio.close();

        double vers = variables[1].data<Mat>().at<double>(0,0);
        if (vers != 1.4) qFatal("This is an old vbb version, we don't mess with that.");

        A = variables[0].data<vector<vector<MatlabIOContainer> > >().at(0);
        objLists = A.at(1).data<vector<MatlabIOContainer> >();

        // start at the first frame (duh!)
        currFrame = 0;
    }

    TemplateList readBlock(bool *done)
    {
        *done = false;
        Template rects(file);
        if (objLists[currFrame].typeEquals<vector<vector<MatlabIOContainer> > >()) {
            vector<vector<MatlabIOContainer> > bbs = objLists[currFrame].data<vector<vector<MatlabIOContainer> > >();
            for (unsigned int i=0; i<bbs.size(); i++) {
                vector<MatlabIOContainer> bb = bbs[i];
                Mat pos = bb[1].data<Mat>();
                double left = pos.at<double>(0,0);
                double top = pos.at<double>(0,1);
                double width = pos.at<double>(0,2);
                double height = pos.at<double>(0,3);
                rects.file.appendRect(QRectF(left, top, width, height));
            }
        }
        TemplateList tl;
        tl.append(rects);
        if (++currFrame == (int)objLists.size()) *done = true;
        return tl;
    }

    void write(const Template &t)
    {
        (void)t; qFatal("Not implemented");
    }

private:
    // this holds a bunch of stuff, maybe we'll use it all later
    vector<MatlabIOContainer> A;
    // this, a field in A, holds bounding boxes for each frame
    vector<MatlabIOContainer> objLists;
    int currFrame;
};

BR_REGISTER(Gallery, vbbGallery)

} // namespace br

#include "gallery/vbb.moc"
