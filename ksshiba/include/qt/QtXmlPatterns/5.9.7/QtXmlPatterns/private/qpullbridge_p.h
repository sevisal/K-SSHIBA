/****************************************************************************
**
** Copyright (C) 2016 The Qt Company Ltd.
** Contact: https://www.qt.io/licensing/
**
** This file is part of the QtXmlPatterns module of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:LGPL$
** Commercial License Usage
** Licensees holding valid commercial Qt licenses may use this file in
** accordance with the commercial license agreement provided with the
** Software or, alternatively, in accordance with the terms contained in
** a written agreement between you and The Qt Company. For licensing terms
** and conditions see https://www.qt.io/terms-conditions. For further
** information use the contact form at https://www.qt.io/contact-us.
**
** GNU Lesser General Public License Usage
** Alternatively, this file may be used under the terms of the GNU Lesser
** General Public License version 3 as published by the Free Software
** Foundation and appearing in the file LICENSE.LGPL3 included in the
** packaging of this file. Please review the following information to
** ensure the GNU Lesser General Public License version 3 requirements
** will be met: https://www.gnu.org/licenses/lgpl-3.0.html.
**
** GNU General Public License Usage
** Alternatively, this file may be used under the terms of the GNU
** General Public License version 2.0 or (at your option) the GNU General
** Public license version 3 or any later version approved by the KDE Free
** Qt Foundation. The licenses are as published by the Free Software
** Foundation and appearing in the file LICENSE.GPL2 and LICENSE.GPL3
** included in the packaging of this file. Please review the following
** information to ensure the GNU General Public License requirements will
** be met: https://www.gnu.org/licenses/gpl-2.0.html and
** https://www.gnu.org/licenses/gpl-3.0.html.
**
** $QT_END_LICENSE$
**
****************************************************************************/

//
//  W A R N I N G
//  -------------
//
// This file is not part of the Qt API.  It exists purely as an
// implementation detail.  This header file may change from version to
// version without notice, or even be removed.
//
// We mean it.

#ifndef PATTERNIST_PULLBRIDGE_P_H
#define PATTERNIST_PULLBRIDGE_P_H

#include <QtCore/QPair>
#include <QtCore/QStack>

#include <private/qabstractxmlforwarditerator_p.h>
#include <private/qabstractxmlpullprovider_p.h>
#include <private/qitem_p.h>

QT_BEGIN_NAMESPACE

namespace QPatternist
{
    class PullBridge : public AbstractXmlPullProvider
    {
    public:
        inline PullBridge(const QXmlNodeModelIndex::Iterator::Ptr &it) : m_current(StartOfInput)
        {
            Q_ASSERT(it);
            m_iterators.push(qMakePair(StartOfInput, it));
        }

        virtual Event next();
        virtual Event current() const;
        virtual QXmlName name() const;
        /**
         * Returns always an empty QVariant.
         */
        virtual QVariant atomicValue() const;
        virtual QString stringValue() const;
        virtual QHash<QXmlName, QString> attributes();
        virtual QHash<QXmlName, QXmlItem> attributeItems();

        QXmlNodeModelIndex index() const;
        QSourceLocation sourceLocation() const;

    private:
        typedef QStack<QPair<Event, QXmlNodeModelIndex::Iterator::Ptr> > IteratorStack;
        IteratorStack      m_iterators;
        QXmlNodeModelIndex m_index;
        Event              m_current;
    };
}

QT_END_NAMESPACE

#endif
