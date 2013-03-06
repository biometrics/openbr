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

#include "NaturalStringCompare.h"
#include <QStringList>

#define INCBUF() { buffer += curr; ++pos; curr = ( pos < string.length() ) ? string[ pos ] : QChar(); }

void ExtractToken( QString & buffer, const QString & string, int & pos, bool & isNumber )
{
	buffer.clear();
	if ( string.isNull() || pos >= string.length() )
		return;

	isNumber = false;
	QChar curr = string[ pos ];
	if ( curr == '-' || curr == '+' || curr.isDigit() )
	{
		if ( curr == '-' || curr == '+' )
			INCBUF();

		if ( !curr.isNull() && curr.isDigit() )
		{
			isNumber = true;
			while ( curr.isDigit() )
				INCBUF();

			if ( curr == '.' )
			{
				INCBUF();
				while ( curr.isDigit() )
					INCBUF();
			}

			if ( !curr.isNull() && curr.toLower() == 'e' )
			{
				INCBUF();
				if ( curr == '-' || curr == '+' )
					INCBUF();

				if ( curr.isNull() || !curr.isDigit() )
					isNumber = false;
				else
					while ( curr.isDigit() )
						INCBUF();
			}
		}
	}

	if ( !isNumber )
	{
		while ( curr != '-' && curr != '+' && !curr.isDigit() && pos < string.length() )
			INCBUF();
	}
}

int NaturalStringCompare( const QString & lhs, const QString & rhs, Qt::CaseSensitivity caseSensitive )
{
	int ii = 0;
	int jj = 0;

	QString lhsBufferQStr;
	QString rhsBufferQStr;

	int retVal = 0;

	// all status values are created on the stack outside the loop to make as fast as possible
	bool lhsNumber = false;
	bool rhsNumber = false;

	double lhsValue = 0.0;
	double rhsValue = 0.0;
	bool ok1;
	bool ok2;

	while ( retVal == 0 && ii < lhs.length() && jj < rhs.length() )
	{
		ExtractToken( lhsBufferQStr, lhs, ii, lhsNumber );
		ExtractToken( rhsBufferQStr, rhs, jj, rhsNumber );

		if ( !lhsNumber && !rhsNumber )
		{
			// both strings curr val is a simple strcmp
			retVal = lhsBufferQStr.compare( rhsBufferQStr, caseSensitive );

			int maxLen = qMin( lhsBufferQStr.length(), rhsBufferQStr.length() );
			QString tmpRight = rhsBufferQStr.left( maxLen );
			QString tmpLeft = lhsBufferQStr.left( maxLen );
			if ( tmpLeft.compare( tmpRight, caseSensitive ) == 0 )
			{
				retVal = lhsBufferQStr.length() - rhsBufferQStr.length();
				if ( retVal )
				{
					QChar nextChar;
					if ( ii < lhs.length() ) // more on the lhs
						nextChar = lhs[ ii ];
					else if ( jj < rhs.length() ) // more on the rhs
						nextChar = rhs[ jj ];

					bool nextIsNum = ( nextChar == '-' || nextChar == '+' || nextChar.isDigit() );

					if ( nextIsNum )
						retVal = -1*retVal;
				}
			}
		}
		else if ( lhsNumber && rhsNumber )
		{
			// both numbers, convert and compare
			lhsValue = lhsBufferQStr.toDouble( &ok1 );
			rhsValue = rhsBufferQStr.toDouble( &ok2 );
			if ( !ok1 || !ok2 )
				retVal = lhsBufferQStr.compare( rhsBufferQStr, caseSensitive );
			else if ( lhsValue > rhsValue )
				retVal = 1;
			else if ( lhsValue < rhsValue )
				retVal = -1;
		}
		else
		{
			// completely arebitrary that a number comes before a string
			retVal = lhsNumber ? -1 : 1;
		}
	}

	if ( retVal != 0 )
		return retVal;
	if ( ii < lhs.length() )
		return -1;
	else if ( jj < rhs.length() )
		return 1;
	else
		return 0;
}

bool NaturalStringCompareLessThan( const QString & lhs, const QString & rhs )
{
	return NaturalStringCompare( lhs, rhs, Qt::CaseSensitive ) < 0;
}

bool NaturalStringCaseInsensitiveCompareLessThan( const QString & lhs, const QString & rhs )
{
	return NaturalStringCompare( lhs, rhs, Qt::CaseInsensitive ) < 0;
}

QStringList NaturalStringSort( const QStringList & list, Qt::CaseSensitivity caseSensitive )
{
	QStringList retVal = list;
	if ( caseSensitive == Qt::CaseSensitive )
		qSort( retVal.begin(), retVal.end(), NaturalStringCompareLessThan );
	else
		qSort( retVal.begin(), retVal.end(), NaturalStringCaseInsensitiveCompareLessThan );
	return retVal;
}


