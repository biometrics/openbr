/*
 * This software was written by people from OnShore Consulting services LLC
 * <info@sabgroup.com> and placed in the public domain.
 *
 * We reserve no legal rights to any of this. You are free to do
 * whatever you want with it. And we make no guarantee or accept
 * any claims on damages as a result of this.
 *
 * If you change the software, please help us and others improve the
 * code by sending your modifications to us. If you choose to do so,
 * your changes will be included under this license, and we will add
 * your name to the list of contributors.
*/

#include <QCoreApplication>
#include <QtTest>
#include <QDebug>
#include "NaturalStringCompare.h"

class CTestNaturalStringCompare : public QObject
{
	Q_OBJECT
private slots:
	void compareString()
	{
		QString str1 = "abcdef";
		QString str2 = "aabc";
		QVERIFY( NaturalStringCompare(str1,str2) > 0 );
		QVERIFY( NaturalStringCompare(str2,str1) < 0 );
		QVERIFY( NaturalStringCompare(str1,str1) == 0 );
		QVERIFY( NaturalStringCompare(str2,str2) == 0 );

		QVERIFY( NaturalStringCompare(str1,str1.left(2)) > 0 );
		QVERIFY( NaturalStringCompare(str1.left(2),str1) < 0 );
	}

	void compareInteger()
	{
		for( int ii = -15; ii <= 15; ii++ )
		{
			QString currVal = QString( "%1" ).arg( ii );
			QString nextVal = QString( "%1" ).arg( ii+1 );

			QVERIFY( NaturalStringCompare(currVal,nextVal) < 0 );
			QVERIFY( NaturalStringCompare( "prefix" + currVal, "prefix" + nextVal) < 0 );
			QVERIFY( NaturalStringCompare( currVal + "postfix", nextVal + "postfix") < 0 );
			QVERIFY( NaturalStringCompare( "prefix" + currVal + "postfix", "prefix" + nextVal + "postfix") < 0 );

			QVERIFY( NaturalStringCompare(nextVal,currVal) > 0 );
			QVERIFY( NaturalStringCompare( "prefix" + nextVal, "prefix" + currVal) > 0 );
			QVERIFY( NaturalStringCompare( nextVal + "postfix", currVal + "postfix") > 0 );
			QVERIFY( NaturalStringCompare( "prefix" + nextVal + "postfix", "prefix" + currVal + "postfix") > 0 );

			QVERIFY( NaturalStringCompare(currVal,currVal) == 0 );
			QVERIFY( NaturalStringCompare( "prefix" + currVal, "prefix" + currVal) == 0 );
			QVERIFY( NaturalStringCompare( currVal + "postfix", currVal + "postfix") == 0 );
			QVERIFY( NaturalStringCompare( "prefix" + currVal + "postfix", "prefix" + currVal + "postfix") == 0 );

			QVERIFY( NaturalStringCompare( "prefix" + currVal + "middle" + currVal, "prefix" + currVal + "middle" + nextVal ) < 0 );
			QVERIFY( NaturalStringCompare( "prefix" + currVal + "middle" + nextVal, "prefix" + currVal + "middle" + currVal ) > 0 );
		}
	}

	void compareDouble()
	{
		for( int ii = -15; ii <= 15; ii++ )
		{
			QString currVal = QString( "%1" ).arg( 1.0*ii - 0.05 );
			QString nextVal = QString( "%1" ).arg( 1.0*ii + 0.05 );

			QVERIFY( NaturalStringCompare(currVal,nextVal) < 0 );
			QVERIFY( NaturalStringCompare( "prefix" + currVal, "prefix" + nextVal) < 0 );
			QVERIFY( NaturalStringCompare( currVal + "postfix", nextVal + "postfix") < 0 );
			QVERIFY( NaturalStringCompare( "prefix" + currVal + "postfix", "prefix" + nextVal + "postfix") < 0 );

			QVERIFY( NaturalStringCompare(nextVal,currVal) > 0 );
			QVERIFY( NaturalStringCompare( "prefix" + nextVal, "prefix" + currVal) > 0 );
			QVERIFY( NaturalStringCompare( nextVal + "postfix", currVal + "postfix") > 0 );
			QVERIFY( NaturalStringCompare( "prefix" + nextVal + "postfix", "prefix" + currVal + "postfix") > 0 );

			QVERIFY( NaturalStringCompare(currVal,currVal) == 0 );
			QVERIFY( NaturalStringCompare( "prefix" + currVal, "prefix" + currVal) == 0 );
			QVERIFY( NaturalStringCompare( currVal + "postfix", currVal + "postfix") == 0 );
			QVERIFY( NaturalStringCompare( "prefix" + currVal + "postfix", "prefix" + currVal + "postfix") == 0 );

			QVERIFY( NaturalStringCompare( "prefix" + currVal + "middle" + currVal, "prefix" + currVal + "middle" + nextVal ) < 0 );
			QVERIFY( NaturalStringCompare( "prefix" + currVal + "middle" + nextVal, "prefix" + currVal + "middle" + currVal ) > 0 );
		}
	}

	void sortStringList()
	{
		QStringList orig;
		for( int ii = 15; ii >= -15; ii-- )
		{
			QString currVal = QString( "%1" ).arg( 1.0*ii - 0.05 );
			orig << "prefix" + currVal + "postfix";
			orig << "Prefix" + currVal + "posTfIx";
			currVal = QString( "%1" ).arg( ii );
			orig << "Prefix" + currVal;
			orig << currVal + "PostFIX";
			orig << currVal + "PostFix";
		}

		QStringList toBeSorted = orig;
		qSort( toBeSorted.begin(), toBeSorted.end(), NaturalStringCompareLessThan );


		int cnt = toBeSorted.count();
		QVERIFY( toBeSorted.count() == 31 * 5 );
		QVERIFY( toBeSorted.front() == "-15PostFIX" );
		QVERIFY( toBeSorted.back() == "prefix14.95postfix" );

		QStringList tmp = NaturalStringSort( orig );
		QVERIFY( tmp.count() == toBeSorted.count() );
		for( int ii = 0; ii < tmp.count(); ++ii )
		{
			QVERIFY( tmp[ ii ] == toBeSorted[ ii ] );
		}

		toBeSorted = orig;
		qSort( toBeSorted.begin(), toBeSorted.end(), NaturalStringCaseInsensitiveCompareLessThan );
		cnt = toBeSorted.count();
		QVERIFY( toBeSorted.count() == 31 * 5 );
		// QVERIFY( toBeSorted.front() == "-15PostFix" ); for case insensitive, PostFix vs PostFIX is non-deterministic which comes firs
		// QVERIFY( toBeSorted.back() == "Prefix15" );

		tmp = NaturalStringSort( orig, Qt::CaseInsensitive );
		QVERIFY( tmp.count() == toBeSorted.count() );
		for( int ii = 0; ii < tmp.count(); ++ii )
		{
			QVERIFY( tmp[ ii ] == toBeSorted[ ii ] );
		}
	}

	void lingfaYangTest()
	{
		// this is actually not what the spec wants, however, the 8 causes the first compare to be
		// ppt/slides/slide vs ppt/slides/slide.xml which means embedded numbers will come first.

		QVERIFY( NaturalStringCompare( "ppt/slides/slide8.xml", "ppt/slides/slide.xml2", Qt::CaseInsensitive ) > 0 );

		QStringList orderedList = QStringList() // from Ligfa Ya email
			<< "[Content_Types].xml"
			<< "_rels/.rels"
			<< "ppt/media/image8.wmf"
			<< "ppt/media/image9.jpeg"
			<< "ppt/media/image10.png"
			<< "ppt/media/image11.gif"
			<< "ppt/slides/_rels/slide9.xml.rels"
			<< "ppt/slides/_rels/slide10.xml.rels"
			<< "ppt/slides/slide.xml" 
			<< "ppt/slides/slide8.xml"
			<< "PPT/SLIDES/SLIDE9.XML"
			<< "ppt/slides/slide10.xml"
			<< "ppt/slides/slide11.xml"
			<< "slide.xml"
			;

		QStringList toBeSorted = QStringList();
		QStringList tmp = orderedList;
		qsrand( QDateTime::currentDateTime().toTime_t() );
		while( !tmp.isEmpty() )
		{
			double randVal = qrand();
			randVal /= RAND_MAX;
			int val = (int)( tmp.count() - 1 ) * randVal;
			toBeSorted << tmp[ val ];
			tmp.removeAt( val );
		}
		tmp = NaturalStringSort( toBeSorted, Qt::CaseInsensitive );
		for( int ii = 0; ii < tmp.count(); ++ii )
		{
			QVERIFY( tmp[ ii ] == orderedList[ ii ] );
		}
	}
	
};

QTEST_MAIN(CTestNaturalStringCompare)
#include "main.moc"
