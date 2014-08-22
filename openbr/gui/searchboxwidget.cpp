#include "searchboxwidget.h"

using namespace br;

SearchBoxWidget::SearchBoxWidget(QWidget *parent) :
    QWidget(parent)
{
    model = new QStringListModel(this);
    completer = new QCompleter(this);
    completer->setCaseSensitivity(Qt::CaseInsensitive);
    searchBar = new QLineEdit(this);

    layout = new QVBoxLayout(this);
    layout->addWidget(searchBar);

    setLayout(layout);

    startIndex = 0;

    connect(searchBar, SIGNAL(returnPressed()), this, SLOT(setIndex()));
}

void SearchBoxWidget::setFiles(br::FileList files)
{
    words.clear();

    foreach (const br::File file, files)
        words.push_back(file.get<QString>("LASTNAME","N/A") + ", " + file.get<QString>("FIRSTNAME", "N/A"));

    model->setStringList(words);
    completer->setModel(model);

    searchBar->setCompleter(completer);
}

void SearchBoxWidget::setIndex()
{
    if (searchBar->text().isEmpty()) return;

    // Get index of currently selected object, starting at the previous searches index
    int index = words.indexOf(searchBar->text(), startIndex);

    // Start from beginning of list if nothing is found
    if (index == -1) {
        index = words.indexOf(searchBar->text(), 0);
        startIndex = index+1;
    }
    else startIndex = index+1;

    if (index != -1) emit newIndex(index);
}
